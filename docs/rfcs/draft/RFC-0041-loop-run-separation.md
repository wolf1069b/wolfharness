---
rfc_id: RFC-0041
title: "Run vs Turn: Separating Session-Level Persistence from Reactive Execution"
status: DRAFT
author: yuchen.liu
reviewers: []
created: 2026-06-27
last_updated: 2026-06-28 (revision 7: SessionController simplification + TurnRunner deletion + deprecation strategy)
decision_date:
related_rfcs:
  - RFC-0029 (Agent Reactivation via Pending Prompt Queue — legacy inject_prompt/queue_prompt)
  - RFC-0037 (Unify Steer and Followup Message Injection — maps to pydantic-ai enqueue)
  - RFC-0021 (Agent Concurrent Execution Safety — per-run context isolation)
related_specs:
  - openspec/changes/introduce-anyio-structured-concurrency/ (CancelScope hierarchy)
  - openspec/changes/structured-work-channel/ (archived — work channel design)
---

# RFC-0041: Run vs Turn — Separating Session-Level Persistence from Reactive Execution

## Overview

AgentPool's current architecture conflates two distinct concepts: the **session-level lifecycle** (long-lived, receives multiple prompts) and the **reactive execution cycle** (single prompt → model → tools → response). This conflation manifests as a 1:1:1 binding between prompt, turn, and `RunHandle`, requiring ~415 lines of compensating complexity (dual queues, auto-resume, re-iteration loops, 4-branch steer/followup) — and ~2500 lines total in the orchestrator layer — to simulate persistent sessions.

This RFC proposes separating these into two orthogonal concepts:

- **Run** (concept): A session-level persistent execution context with idle/running/done states. One Run per session. Protocol-agnostic. The existing `RunHandle` class is **restructured** (not renamed) to implement this concept — same class name, evolved internals.
- **Turn**: A single reactive cycle (prompt → response). N Turns per Run, executed serially. Agent-type-specific implementation.

Native agent Turns become thin wrappers over pydantic-ai's `agent.iter()` → `next(node)` → `End` cycle, reusing pydantic-ai's `PendingMessageDrainCapability` for in-turn message drain. Non-native (ACP) Turns wrap a single `session/prompt` → stream → complete cycle. Both share the same Run for idle management, steer/followup routing, and event publishing.

Run lifecycle is managed explicitly via `async with` and `run.close()` — not via timeout. Timeout is a caller-side policy, not a Run mechanism. Standalone execution (without SessionPool) is a first-class use case: idle/wait/steer does not require SessionPool infrastructure.

The expected outcome is ~415 lines of compensating complexity eliminated, with the orchestrator layer reduced from ~2500 lines to ~1000 lines. Native agent Turns reach ~80 lines (including event mapping delegation and exception handling) by delegating to pydantic-ai primitives.

## Table of Contents

- Background & Context
- [Problem Statement](#problem-statement)
- Goals & Non-Goals
- [Evaluation Criteria](#evaluation-criteria)
- [Options Analysis](#options-analysis)
- [Recommendation](#recommendation)
- [Technical Design](#technical-design)
- Security Considerations
- Implementation Plan
- [Open Questions](#open-questions)
- Decision Record
- [References](#references)

---

## Background & Context

### Current State

AgentPool's agent execution stack has three layers, each with agent-type-specific branching:

```
Protocol Server (ACP/OpenCode/AG-UI/OpenAI API)
    ↓
TurnRunner (run_loop, _run_turn_unlocked, steer, followup)
    ↓               ↓
RunExecutor       Manual ACP queue
(native only)     (non-native only)
```

**Key components and their responsibilities:**

| Component | File | Lines | Responsibility |
|-----------|------|-------|----------------|
| `RunHandle` | `orchestrator/run.py` | 150 | Per-run lifecycle (pending→running→completed/failed/checkpointed), `complete_event`, cancel |
| `RunExecutor` | `orchestrator/run_executor.py` | 440 | Native agent turn execution + re-iteration loop for steer messages |
| `TurnRunner` | `orchestrator/core.py` L1751-2605 | ~850 | Turn execution, dual queues, steer/followup with native/non-native branching, auto-resume |
| `SessionController` | `orchestrator/core.py` L774-1700 | ~900 | Session management, `receive_request()`, run creation |
| `PromptInjectionManager` | `agents/prompt_injection.py` | 143 | Tool-result augmentation + follow-up prompt queuing (non-native) |

**Current v1 model: 1 prompt = 1 turn = 1 RunHandle (1:1:1)**

When a prompt arrives at an idle session:
1. `SessionController.receive_request()` creates a `RunHandle`
2. `TurnRunner.run_loop()` starts, acquires `session.turn_lock`
3. `_run_turn_unlocked()` calls `RunExecutor.execute()` (native) or direct ACP prompt (non-native)
4. Turn completes → `RunHandle.complete()` → `complete_event.set()`
5. `run_loop()` exits, `session.current_run_id` cleared

If messages arrive while busy, they route through `steer()` or `followup()`, which have **4 branches** each (native×active, native×idle, non-native×active, non-native×idle). The idle branches either delegate to `receive_request()` (creating a new RunHandle) or store in `_post_turn_injections`/`_post_turn_prompts` dicts and trigger `_trigger_auto_resume()`.

### pydantic-ai Primitives Available

pydantic-ai (installed at `.venv/lib/python3.13/site-packages/pydantic_ai/`) provides several primitives relevant to this design:

| Primitive | Location | Relevance |
|-----------|----------|-----------|
| `AgentRun` | `run.py` L33-475 | Wraps a `GraphRun`, provides `next(node)` (fires hooks) and `enqueue()` |
| `PendingMessageDrainCapability` | `capabilities/_pending_messages.py` L60-160 | Auto-injected outermost capability; drains `'asap'` at `before_model_request`, drains `'when_idle'` at `after_node_run` and redirects `End → ModelRequestNode` |
| `Agent.iter()` | `agent/__init__.py` L1041+ | Creates `AgentRun` context manager for a single reactive cycle |
| `GraphAgentState` | `_agent_graph.py` L117-175 | Holds `message_history`, `pending_messages`, `run_step` — all state needed to resume |
| `BaseNode` | `pydantic_graph/basenode.py` L37-141 | Abstract node; custom nodes can be created |
| `End` | `pydantic_graph/basenode.py` L143-167 | Terminal node; can be intercepted by `after_node_run` hook |
| Capabilities system | `capabilities/abstract.py` L134-899 | 18 lifecycle hooks; `CapabilityOrdering` for topological sort |

**Key finding**: pydantic-ai has no native pause/idle/resume mechanism. `AgentRun` terminates at `End`. However, `PendingMessageDrainCapability.after_node_run()` already demonstrates the `End → ModelRequestNode` redirect pattern, which is the foundation for "don't terminate yet" semantics.

**Critical iteration requirement**: `AgentRun.next(node)` must be used instead of bare `async for node in agent_run` — the latter skips all capability hooks including `PendingMessageDrainCapability`'s drain logic. The current `RunExecutor` already uses `next(node)` correctly.

**Note on naming**: pydantic-ai's `AgentRun` is an internal implementation detail of a single reactive cycle. In this RFC, we call the reactive cycle a **Turn** and use `agent_run` as a local variable name inside `NativeTurn.execute()`. There is no naming conflict because `AgentRun` is never exposed to wolfharness users.

### anyio Primitives Available

| Primitive | Reusable? | Relevance to idle/wake |
|-----------|-----------|------------------------|
| `Condition` | Yes — full wait/notify cycle | Primary candidate for idle/wake signaling |
| `Event` | No — no `clear()` method | Unsuitable for multi-cycle idle; `asyncio.Event` used instead (has `clear()`). Codebase already uses `asyncio.Event` for `RunHandle.complete_event`. NOTE: `asyncio.Event` is asyncio-backend-only; trio would need `anyio.Condition`. |
| `CancelScope` | No — single `with` block | Per-run scope; `shield=True` for critical sections |
| `TaskGroup` | Group-level: yes | Already used via `_session_task_groups` (never exits) |
| `CapacityLimiter` | Token-based | Could replace manual `_max_concurrent_runs` counting |

### Historical Context

- **RFC-0029** (2026-04-26): Introduced `inject_prompt()`/`queue_prompt()` with `asyncio.Event` notification for agent reactivation between `run_stream()` calls. This was the first attempt at "idle" semantics — the agent provides the signal, the caller provides the reactivation loop.
- **RFC-0037** (2026-06-15): Proposed unifying `steer()`/`followup()` by mapping to pydantic-ai's `enqueue()` for native agents. Recognized the dual-system redundancy but kept the per-RunHandle lifecycle.
- **`introduce-anyio-structured-concurrency`** OpenSpec change (completed): Established the CancelScope hierarchy (pool→session→agent run→subagent run). `RunHandle._cleanup_run()` sets `complete_event` in `anyio.CancelScope(shield=True)`.
- **ACP v2 Prompt Lifecycle RFD** (by @benbrandt): Proposes `session/prompt` returning on accept (fire-and-forget), with turn lifecycle communicated via `state_change` notifications. **PR #1261** (`session/inject` by @kennethsinder) adds `mode: "queue" | "steer"` — directly mapping to wolfharness's `followup()` / `steer()`.

### Glossary

| Term | Definition |
|------|------------|
| **Run** | Session-level persistent execution context. Survives across multiple Turns. Has idle/running/done states. Protocol-agnostic. Implemented by the existing `RunHandle` class, restructured. |
| **Turn** | Single reactive cycle: prompt → model → tools → response. Agent-type-specific. Bound to a RunHandle. |
| **RunHandle** | Existing per-run lifecycle handle class (`orchestrator/run.py`). In the proposed design, **restructured** (not renamed) to absorb Run semantics: idle/running/done states, message queue, steer/followup routing, `async with` lifecycle. Class name preserved for API stability. |
| **Idle** | RunHandle state between Turns. No active model iteration. Waiting for new messages via `asyncio.Event.wait()`. |
| **Steer** | Inject a message into an active turn that the model sees at the earliest opportunity (before next model call). Maps to ACP v2 `session/inject mode: "steer"`. |
| **Followup** | Queue a message to be processed after the current Turn completes. Maps to ACP v2 `session/inject mode: "queue"`. |
| **`PendingMessageDrainCapability`** | pydantic-ai auto-injected capability. Handles `asap`/`when_idle` message priorities within a single `AgentRun.iter()` cycle. |

---

## Problem Statement

### Problem 1: Conceptual Conflation

The 1:1:1 binding (prompt = turn = RunHandle) forces the system to simulate session persistence through destruction-and-recreation:

```
prompt → RunHandle #1 → complete → destroy
         ↓ (message arrives while idle)
         _post_turn_injections stores message
         _trigger_auto_resume spawns task
         → new RunHandle #2 → complete → destroy
```

This pattern appears in `TurnRunner._trigger_auto_resume()` (L2561-2605), `_process_queued_work()` (L2489-2559), and the RunExecutor re-iteration loop (L389-414). Each is a patch over the missing "run persists between turns" primitive.

### Problem 2: Agent-Type Branching

`TurnRunner.steer()` and `followup()` each have 4 branches:

```
steer():
  native + active   → agent_run.enqueue(priority="asap")
  native + idle     → receive_request(priority="steer")  [creates new RunHandle]
  non-native + active → injection_manager.inject()
  non-native + idle   → _post_turn_injections + _trigger_auto_resume
```

This branching propagates to `run_loop()`, `_run_turn_unlocked()`, and `close_session()`, creating agent-type-specific code paths throughout the orchestrator.

### Problem 3: Compensating Complexity

| Component | Lines | Exists Because |
|-----------|-------|----------------|
| RunExecutor re-iteration loop (L389-414) | ~25 | Turn terminates at End; need to check for queued steer messages |
| `_post_turn_injections` / `_post_turn_prompts` | ~40 | RunHandle destroyed; need dict to store messages between runs |
| `_trigger_auto_resume()` | ~45 | RunHandle destroyed; need to spawn new task to process queued messages |
| `_process_queued_work()` | ~70 | RunHandle destroyed; need loop to drain queued prompts |
| `steer()`/`followup()` native/non-native branching | ~80 | Different queue mechanisms for native (enqueue) vs non-native (dict) |
| `_run_turn_unlocked()` finally block | ~55 | Must clean up RunHandle, clear `current_run_id`, reset ContextVars |
| `RunExecutor.execute()` task_group + cancel handling | ~100 | Wraps pydantic-ai iteration for CancelScope safety |

**Total: ~415 lines of compensating complexity.**

### Evidence

- `TurnRunner` class spans ~850 lines (core.py L1751-2605), with the majority handling edge cases around run lifecycle transitions
- `RunExecutor.execute()` is 440 lines, of which only ~170 are the actual pydantic-ai iteration loop; the rest is re-iteration, cancel handling, and event mapping
- ACP v2's `session/inject` (PR #1261) maps directly to `steer()`/`followup()`, but the current 4-branch implementation makes this mapping non-trivial
- The `introduce-anyio-structured-concurrency` OpenSpec change (69/69 tasks completed) established CancelScope hierarchy but did not address the Run/Turn separation

### Impact of Inaction

- **Cost**: Continued maintenance of ~415 lines of compensating complexity; each new feature (e.g., ACP v2 support) must navigate 4-branch agent-type dispatching
- **Risk**: ACP v2 migration requires decoupling prompt from turn lifecycle; without Run/Turn separation, this requires additional patches on top of the existing patch layer
- **Opportunity**: Native agent Turns could be ~80 lines (thin pydantic-ai wrapper with event mapping delegation) instead of 440 lines, reducing the surface area for bugs and enabling faster iteration

---

## Goals & Non-Goals

### Goals (In Scope)

1. **Separate Run and Turn concepts**: Run = session-level persistent context (idle/running/done); Turn = single reactive cycle (agent-type-specific)
2. **Unify steer/followup**: Single implementation, no native/non-native branching at the Run level
3. **Thin native Turn**: Native agent Turn implementation delegates to pydantic-ai `iter()`/`next(node)` with event mapping delegation (~80 lines including exception handling and terminal tool support)
4. **Explicit lifecycle management**: Run lifecycle via `async with` and `run.close()` — no timeout-based stop. Timeout is caller's policy, not Run's mechanism.
5. **Standalone execution**: `agent.run()` works without SessionPool — idle/wait/steer are Run primitives, not orchestrator infrastructure
6. **Multi-protocol subscription**: Multiple protocol servers can subscribe to the same Run's event stream via EventBus (already supported, documented here)
7. **Eliminate compensating complexity**: Remove re-iteration loop, dual queues, auto-resume, 4-branch steer/followup
 8. **ACP v2 alignment**: RunHandle's idle mechanism naturally maps to v2's `state_change` notifications and `session/inject` modes

### Non-Goals (Out of Scope)

1. **ACP v2 protocol implementation**: This RFC designs the runtime architecture; protocol-level v2 migration is a separate effort
2. **Non-native agent migration to pydantic-ai**: ACP agents will continue using JSON-RPC; only the Run layer is unified
3. **Subagent/spawn-session changes**: The Run/Turn separation affects top-level sessions; subagent spawning is handled by existing graph architecture
4. **Message history serialization**: Persistent storage of `message_history` across Run idle periods is a storage-layer concern
5. **Graph-based team execution**: Team orchestration via pydantic-graph is orthogonal to the Run/Turn separation
6. **Multi-server / distributed execution**: Run state is in-process. Distributed Run (shared state across processes) is future work documented but not implemented in this RFC.

### Success Criteria

- [ ] `agent.run()` produces identical behavior to current `RunExecutor.execute()` for single-turn usage
- [ ] `agent.run()` supports idle → wake → next Turn with < 5ms wake latency (measured from `idle_event.set()` to first event yield, warm cache, no model loading)
- [ ] Native agent Turn implementation is ≤ 80 lines (including event mapping delegation and exception handling)
- [ ] `steer()` and `followup()` have zero agent-type branching at the Run level
- [ ] `_post_turn_injections`, `_post_turn_prompts`, `_trigger_auto_resume`, and RunExecutor re-iteration loop are deleted
- [ ] Standalone `agent.run()` (no SessionPool) supports idle/wake/steer without orchestrator infrastructure
- [ ] All existing tests pass without modification to test assertions

---

## Evaluation Criteria

| Criterion | Weight | Description | Minimum Threshold |
|-----------|--------|-------------|-------------------|
| Complexity reduction | High | Net lines removed from orchestrator; cyclomatic complexity of steer/followup | ≥ 300 net lines removed; steer/followup ≤ 2 branches each |
| pydantic-ai alignment | High | Degree to which native Turn delegates to pydantic-ai primitives without wrapping | Native Turn ≤ 80 lines; no custom pydantic-ai nodes or capabilities |
| v1 compatibility | High | `agent.run()` behavior matches existing single-turn execution | All existing tests pass |
| ACP v2 readiness | Medium | Run idle mechanism maps directly to v2 `state_change` + `session/inject` | Idle/wake maps to v2 without additional adapter layer |
| Migration risk | Medium | Ability to migrate incrementally without breaking existing protocol servers | Phase 1 (native only) ships independently |
| Resource efficiency | Low | Memory/CPU overhead of persistent RunHandle vs destroy/recreate | Idle RunHandle memory < 2× current per-turn RunHandle |
| Standalone usability | Medium | `agent.run()` works without SessionPool for idle/wake/steer | No SessionPool import required for standalone |

---

## Options Analysis

### Option 1: Run/Turn Separation with `async with agent.run()` (Recommended)

**Description**

Introduce two orthogonal concepts:

- **`RunHandle`** (restructured): Session-level persistent execution context. The existing `RunHandle` class is restructured (not renamed) to absorb Run semantics. Owns `idle_event` (reusable `asyncio.Event` with `clear()`), `message_queue`, and the `while True` idle/turn cycle. Protocol-agnostic. One per session. Lifecycle managed via `async with` and `close()`. Class name preserved for API stability — existing callers (`close_session()`, `cancel_run()`, `SessionPool._runs`, protocol servers) require no rename.
- **`Turn`** (abstract): Single reactive cycle. Agent-type-specific. `execute()` yields `RichAgentStreamEvent` and returns updated `message_history`.

`agent.run(prompt)` returns a `RunHandle` object that is both an async context manager and an async iterator:

```python
# v1: single Turn (exits async with after first Turn)
async with agent.run("prompt") as run:
    async for event in run.start("prompt"):
        ...

# v2: persistent (idle between Turns, steer from separate task)
# start() is called ONCE. The consumer stays in async for across all Turns.
# steer()/followup() are called from a SEPARATE task (protocol server, etc.)
async with agent.run("prompt") as run:
    async for event in run.start("prompt"):
        ...
        # Turn 1 events flow here...
        # After Turn 1, start() enters idle (blocks on idle_event.wait())
        # A separate task calls run.steer("add tests") to wake the Run
        # Turn 2 events continue flowing through the same async for...
# exit async with → run.close()
```

No `idle_timeout` parameter. Run waits indefinitely until woken by `steer()`, `followup()`, or `close()`. Callers who want timeout wrap with `anyio.move_on_after(N)` or configure session-level policy in SessionPool.

Native `Turn` wraps a single `agentlet.iter()` → `next(node)` → `End` cycle (~80 lines including event mapping delegation and exception handling). Non-native `Turn` wraps a single ACP `session/prompt` → stream → complete cycle (~30 lines).

Steer/followup are unified on `RunHandle`:
- `steer(message)`: If idle → `wake(message)`. If running + native → `agent_run.enqueue(priority="asap")`. If running + non-native → append to `message_queue`.
- `followup(message)`: Always append to `message_queue`. If idle → `wake()`.

**Advantages**

- Eliminates ~415 lines of compensating complexity (re-iteration loop, dual queues, auto-resume, 4-branch steer/followup)
- Native Turn reaches ~80 lines by delegating to pydantic-ai `iter()`/`next(node)` with event mapping extracted to shared `EventMapper`
- `agent.run()` provides zero-migration v1 compatibility (single Turn via `async with`)
- RunHandle's idle/wake mechanism maps directly to ACP v2's `state_change` + `session/inject` without adapter layer
- Steer/followup unified — zero agent-type branching at Run level
- `asyncio.Event` with `clear()` provides reusable multi-cycle idle signaling (no `anyio.Condition` complexity needed)
- **Standalone execution**: `agent.run()` works without SessionPool — idle/wait/steer are Run primitives
- **Explicit lifecycle**: No timeout guessing — `async with` + `close()` is clear and deterministic
- **Multi-protocol**: EventBus already supports multiple subscribers per session; Run/Turn separation makes Run state visible to all protocol servers

**Disadvantages**

- `RunHandle` semantics change: from per-turn to per-session. Callers that expect `complete_event` after each turn must adapt.
- `close_session()` must handle idle RunHandle state: force-wake + cancel instead of waiting for `complete_event` (deadlock risk if not addressed)
- `session.current_run_id` semantics change: set while RunHandle is alive (including idle), not just during active Turn
- TTL-based session cleanup (`_cleanup_expired_sessions`) must distinguish idle from running to avoid skipping permanently-idle sessions
- Two abstract methods to implement (`create_turn` for each agent type) instead of one `execute()` — minor interface expansion. Event mapping code (~170 lines) is extracted to `EventMapper` rather than eliminated, so total reduction is ~51% not ~67%.
- `RunHandle` must support both `async with` and `async for` — slightly more complex than a plain async generator

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Complexity reduction | 5/5 | ~415 lines of compensating complexity removed; steer/followup → 1-2 branches each. Event mapping extracted to shared `EventMapper` (~180 lines) rather than eliminated |
| pydantic-ai alignment | 5/5 | Native Turn = thin `iter()` wrapper (~80 lines with event mapping delegation), no custom nodes/capabilities |
| v1 compatibility | 4/5 | `agent.run()` matches; `RunHandle` semantics change requires caller adaptation |
| ACP v2 readiness | 5/5 | Idle/wake maps directly to `state_change`/`session/inject` |
| Migration risk | 3/5 | Phase 1 (native only) is self-contained; `RunHandle` semantics change is the main risk |
| Resource efficiency | 4/5 | Persistent Run holds agent + message_history in memory; negligible overhead vs destroy/recreate |
| Standalone usability | 5/5 | No SessionPool required for idle/wake/steer; `async with agent.run()` is self-contained |

**Effort Estimate**

- Complexity: Medium
- Resources: 1 engineer, 2-3 weeks
- Dependencies: `introduce-anyio-structured-concurrency` (completed), pydantic-ai `iter()`/`next(node)` API stable

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `close_session()` deadlock on idle RunHandle | Medium | High | Check `RunStatus.idle` before waiting on `complete_event`; force-wake if idle |
| `RunHandle` API breakage for existing callers | Medium | Low | `RunHandle` class name preserved; internals restructured. Existing attribute access (`run_id`, `complete_event`, `status`) remains compatible. New attributes (`idle_event`, `message_queue`) are additive. |
| Subagent spawn-session interaction | Low | Medium | Subagent spawning uses existing graph architecture; Run/Turn separation is top-level only |
| `turn_lock` held during idle | Medium | Low | By design (serializes turns); `close_session()` calls `cancel()` to wake |

---

### Option 2: IdleNode + IdleStateCapability (pydantic-ai layer)

**Description**

Implement idle within pydantic-ai's graph model:

- Create a custom `IdleNode(BaseNode)` whose `run()` method blocks on `asyncio.Event.wait()` until new messages arrive, then returns `ModelRequestNode`.
- Create `IdleStateCapability(AbstractCapability)` with `after_node_run` hook that intercepts `End` (when queue is empty) and returns `IdleNode` instead.
- Wire `IdleStateCapability` into the capability chain with `CapabilityOrdering(wrapped_by=[PendingMessageDrainCapability])` so it sits inside the drain.
- Wrap `AgentRun.enqueue()` to trigger `idle_event.set()` on message arrival.

The agent run never exits `async with agent.iter()` — it loops within the graph between `IdleNode` and `ModelRequestNode`.

**Advantages**

- Idle lives entirely within pydantic-ai's graph model — all capability hooks fire correctly
- `after_node_run` redirect is an established pattern (`PendingMessageDrainCapability` already uses it)
- No wolfharness-level Run concept needed — the graph itself persists

**Disadvantages**

- Requires custom pydantic-ai nodes and capabilities — violates "thin wrapper" principle
- `AgentRun` must stay within `async with agent.iter()` context during idle, holding graph resources
- `enqueue()` must be wrapped to trigger wake-up — fragile monkey-patch
- Harder to implement for non-native agents (ACP has no graph model; would need a parallel mechanism)
- pydantic-ai API changes could break custom nodes/capabilities
- `IdleNode.run()` blocking on `Event.wait()` ties up a graph node indefinitely — unclear if pydantic-graph supports this safely

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Complexity reduction | 3/5 | Removes re-iteration loop but adds IdleNode + Capability + enqueue wrapper |
| pydantic-ai alignment | 2/5 | Extends pydantic-ai with custom nodes/capabilities rather than using existing API |
| v1 compatibility | 3/5 | `agent.run()` would need special mode; `RunHandle` lifecycle changes |
| ACP v2 readiness | 2/5 | Non-native agents can't use graph-based IdleNode; need separate mechanism |
| Migration risk | 2/5 | Custom pydantic-ai extensions risk breakage on upstream updates |
| Resource efficiency | 2/5 | Graph resources held during idle; `AgentRun` context not released |
| Standalone usability | 2/5 | Requires pydantic-ai graph extensions; no standalone benefit |

**Effort Estimate**

- Complexity: Medium-High
- Resources: 1 engineer, 3-4 weeks
- Dependencies: Deep understanding of pydantic-ai capability ordering and graph internals

---

### Option 3: Status Quo with Incremental Patches

**Description**

Keep the current architecture and address issues incrementally:

- RFC-0037 unifies steer/followup by mapping to pydantic-ai `enqueue()` for native agents
- Add `RunStatus.idle` to `RunHandle` without separating Run/Turn concepts
- Modify `_trigger_auto_resume()` to reuse existing `RunHandle` instead of creating new ones
- Add idle timeout to `run_loop()` that waits before exiting

**Advantages**

- Minimal architectural change — existing code paths preserved
- Lowest migration risk — each patch is small and self-contained
- No new abstractions introduced

**Disadvantages**

- Does not address the root cause (1:1:1 binding)
- Compensating complexity remains (~415 lines)
- 4-branch steer/followup remains, just with an additional `idle` status check
- ACP v2 migration still requires significant additional work
- Each future feature must navigate the existing patch layer
- No standalone execution path — idle/wait/steer requires SessionPool + TurnRunner

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Complexity reduction | 1/5 | Adds code (idle status checks) rather than removing |
| pydantic-ai alignment | 2/5 | Uses `enqueue()` but still wraps in RunExecutor re-iteration loop |
| v1 compatibility | 5/5 | No API changes |
| ACP v2 readiness | 1/5 | Still requires adapter layer for v2 state_change/inject |
| Migration risk | 5/5 | Minimal — incremental patches |
| Resource efficiency | 3/5 | Same as current; no improvement |
| Standalone usability | 1/5 | No standalone idle; requires full orchestrator stack |

---

### Options Comparison Summary

| Criterion | Option 1: Run/Turn | Option 2: IdleNode | Option 3: Status Quo |
|-----------|-------------------|-------------------|---------------------|
| Complexity reduction | 5/5 | 3/5 | 1/5 |
| pydantic-ai alignment | 5/5 | 2/5 | 2/5 |
| v1 compatibility | 4/5 | 3/5 | 5/5 |
| ACP v2 readiness | 5/5 | 2/5 | 1/5 |
| Migration risk | 3/5 | 2/5 | 5/5 |
| Resource efficiency | 4/5 | 2/5 | 3/5 |
| Standalone usability | 5/5 | 2/5 | 1/5 |
| **Overall** | **31/35** | **16/35** | **18/35** |

---

## Recommendation

### Recommended Option

**Option 1: Run/Turn Separation with `async with agent.run()`**

### Justification

Option 1 scores highest across all criteria, with particular strength in complexity reduction (5/5), pydantic-ai alignment (5/5), ACP v2 readiness (5/5), and standalone usability (5/5). The key insight driving this recommendation is that idle occurs **between** reactive cycles, not **within** them. This means:

1. Each Turn is a complete, clean `agent.iter()` → `End` cycle — pydantic-ai's `PendingMessageDrainCapability` handles all in-turn message drain naturally
2. The Run's only job is "should I start another Turn?" — a simple `while True` with `idle_event.wait()`
3. No custom pydantic-ai nodes or capabilities are needed — the existing `iter()`/`next(node)`/`End` API is sufficient
4. Lifecycle is explicit via `async with` — no timeout guessing, no resource leaks from forgotten timeouts
5. Standalone execution is a first-class use case — no SessionPool required for idle/wake/steer

Option 2 (IdleNode) scores lower because it pushes idle semantics into pydantic-ai's graph model, requiring custom extensions that are fragile, non-portable, and impossible to apply to non-native agents.

Option 3 (status quo) has the lowest migration risk but fails to address the root cause, leaving ACP v2 migration blocked and standalone execution unsupported.

### Accepted Trade-offs

1. **`RunHandle` semantics change**: `RunHandle` is restructured from per-turn to per-session (session-level Run). Callers that depend on `complete_event` firing after each turn must adapt to check `RunStatus`. Acceptable because the number of such callers is small (`close_session`, `cancel_run`, `SessionPool._cleanup_expired_sessions`). Class name is preserved — no import changes needed.

2. **`turn_lock` held during idle**: The RunHandle holds `session.turn_lock` while idle, preventing other turns from starting. This is the desired behavior (serializes turns within a session) but means `close_session()` must force-wake the RunHandle rather than waiting for natural exit. Acceptable because `close_session()` already has a 30s timeout + cancel fallback.

3. **Two-phase migration**: Native agents migrate first (Phase 1), non-native agents later (Phase 2). During the interim, `TurnRunner` must support both old and new paths. Acceptable because Phase 1 is self-contained and Phase 2 is additive (replacing ACP's `_run_turn_unlocked` path).

4. **No timeout-based stop**: RunHandle waits indefinitely until woken. This is by design — timeout is a policy decision that belongs to the caller, not the RunHandle. Callers wrap with `anyio.move_on_after(N)` or configure SessionPool session-level idle policy.

### Conditions

- Phase 1 must not break any existing protocol server (ACP, OpenCode, AG-UI, OpenAI API)
- `agent.run()` must be a drop-in replacement for `RunExecutor.execute()` in single-turn scenarios
- All tests in `tests/orchestrator/` and `tests/agents/` must pass without assertion changes

---

## Technical Design

### Architecture Overview

```
Session
  └─ RunHandle (protocol-agnostic, session-level persistent)
       ├─ Turn #1 (NativeTurn: pydantic-ai iter() | ACPTurn: session/prompt)
       │   → End / turn complete
       ├─ [idle: idle_event.wait() — no timeout, waits indefinitely]
       ├─ Turn #2
       │   → End / turn complete
       ├─ [idle: idle_event.wait()]
       └─ ... → close() / cancel() → RunHandle done
```

```
┌──────────────────┐
│ Protocol Server  │  (ACP / OpenCode / AG-UI / OpenAI API)
│                  │
│  receive_request │──────► RunHandle.start()
│  steer / followup│──────► RunHandle.steer() / .followup()
│  close_session   │──────► RunHandle.close()
└──────────────────┘
                           ┌─────────────────────────────────┐
                           │ RunHandle (protocol-agnostic)     │
                           │                                  │
                           │  async with run:                 │
                           │    while True:                   │
                           │      turn = agent.create_turn()  │
                           │      async for event in turn:    │
                           │        event_bus.publish(event)  │
                           │      if cancelled: break         │
                           │      if queued_msgs: continue    │
                           │      # idle                      │
                           │      idle_event.wait()           │
                           │      if closing: break           │
                           └────────┬─────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          ┌─────────────────┐             ┌─────────────────┐
          │ NativeTurn      │             │ ACPTurn         │
          │                 │             │                 │
          │ agentlet.iter() │             │ client.prompt() │
          │ next(node) loop │             │ stream events   │
          │ → End           │             │ → complete      │
          └─────────────────┘             └─────────────────┘
```

### Key Components

#### RunHandle (restructured)

- **Responsibility**: Session-level persistent execution. Owns idle/turn cycle, message queue, steer/followup routing. Restructured from existing `RunHandle` class — same class name, evolved internals.
- **Interfaces**: `start()` (async generator), `steer()`, `followup()`, `close()`, `cancel()`
- **State**: `RunStatus` (idle/running/done), `idle_event` (reusable `asyncio.Event`), `message_queue` (list), `_closing` (bool)
- **Lifecycle**: `async with` context manager + `AsyncIterator`
- **Extensibility note**: The `idle_event`, `message_queue`, and `_status` fields are designed as swappable primitives for future multi-server support (see "Multi-Server Future Work"). Replacing `asyncio.Event` with a distributed signal and `list[str]` with a distributed queue would not require changes to `start()` / `steer()` / `followup()` logic.

```python
class RunStatus(Enum):
    idle = auto()
    running = auto()
    done = auto()

class RunHandle:
    """Restructured RunHandle: session-level persistent execution context.

    Evolved from per-turn handle to per-session Run. Class name preserved
    for API stability — existing callers (close_session, cancel_run,
    SessionPool._runs, protocol servers) require no import changes.

    Extensibility: _idle_event and _message_queue are designed as swappable
    primitives for future multi-server support. Replacing them with distributed
    equivalents (Redis pub/sub, Redis Stream) would not change the start()/
    steer()/followup() control flow.
    """

    def __init__(
        self,
        agent: BaseAgent,
        run_ctx: AgentRunContext,
        event_bus: EventBus,
        session: SessionState,
    ):
        self._agent = agent
        self._run_ctx = run_ctx
        self._event_bus = event_bus
        self._session = session
        self._status: RunStatus = RunStatus.idle
        self._closing: bool = False
        # NOTE: asyncio.Event (not anyio.Event) because anyio.Event lacks clear()
        # in the installed version. This is consistent with RunHandle.complete_event
        # which also uses asyncio.Event. The codebase targets asyncio backend only.
        self._idle_event: asyncio.Event = asyncio.Event()
        self._message_queue: list[str] = []
        self._message_history: list[ModelMessage] = []
        # Initialize _run_handle on run_ctx so steer() can access active_agent_run
        # RunHandle acts as the session-level handle (restructured from per-turn)
        # M5: Use getattr for AGENT_TYPE safety (matches core.py pattern)
        self._run_ctx._run_handle = RunHandle(
            run_id=uuid4().hex,
            session_id=session.session_id,
            agent_type=getattr(agent, "AGENT_TYPE", "native"),
            run_ctx=run_ctx,
        )

    async def start(self, initial_prompt: str | list[str]) -> AsyncGenerator[RichAgentStreamEvent, None]:
        """Run the execution loop: execute Turns, idle between them, until close/cancel.

        This is an async generator. Yields RichAgentStreamEvent from each Turn.
        Enters idle between Turns, waiting for steer/followup to wake.
        Exits when close() or cancel() is called, or when no messages remain
        after a Turn and _closing is True.

        start() is called ONCE. steer()/followup() come from separate tasks
        (protocol servers, background tasks). The consumer stays in the
        async for loop across all Turns.
        """
        current_prompts: str | list[str] = initial_prompt
        # try/finally ensures complete_event is set even if CancelledError
        # propagates from close_session() scope.cancel() cascade.
        try:
            async with self._session.turn_lock:
                while True:
                    if self._run_ctx.cancelled or self._closing:
                        break

                    self._status = RunStatus.running
                    turn = self._agent.create_turn(
                        prompts=current_prompts,
                        run_ctx=self._run_ctx,
                        message_history=self._message_history,
                    )
                    # Publish lifecycle start event
                    await self._event_bus.publish(
                        self._run_ctx.session_id,
                        RunStartedEvent(
                            run_id=self._run_ctx._run_handle.run_id,
                            agent_name=self._agent.name,
                            session_id=self._run_ctx.session_id,
                        ),
                    )
                    run_failed = False
                    try:
                        async for event in turn.execute():
                            if self._run_ctx.cancelled:
                                break
                            await self._event_bus.publish(self._run_ctx.session_id, event)
                            yield event
                    except RunAbortedError:
                        self._run_ctx.cancelled = True
                        break
                    except UndrainedPendingMessagesError:
                        logger.warning("Undrained pending messages at Turn completion")
                        run_failed = True
                    except asyncio.CancelledError:
                        self._run_ctx.cancelled = True
                        raise
                    except Exception as exc:
                        # C4: Use correct field name 'message' (not 'error'),
                        # include agent_name
                        logger.exception("Turn failed with unexpected error: %s", exc)
                        await self._event_bus.publish(
                            self._run_ctx.session_id,
                            RunErrorEvent(
                                message=str(exc),
                                agent_name=self._agent.name,
                                run_id=self._run_ctx._run_handle.run_id,
                            ),
                        )
                        run_failed = True
                    finally:
                        self._message_history = turn.message_history

                    if self._run_ctx.cancelled:
                        break

                    # Publish lifecycle complete event
                    if run_failed:
                        final_msg = turn.final_message if turn._final_message is not None else (
                            ChatMessage(role="assistant", content="[Run failed]")
                        )
                    else:
                        final_msg = turn.final_message
                    await self._event_bus.publish(
                        self._run_ctx.session_id,
                        StreamCompleteEvent(
                            message=final_msg,
                            cancelled=self._run_ctx.cancelled,
                            session_id=self._run_ctx.session_id,
                        ),
                    )

                    if run_failed:
                        break

                    # Check for subagent completions (child_done_events)
                    if self._run_ctx.child_done_events:
                        for event in self._run_ctx.child_done_events:
                            await event.wait()
                        steer_msgs = self._run_ctx.queued_steer_messages.copy()
                        self._run_ctx.queued_steer_messages.clear()
                        if steer_msgs:
                            current_prompts = steer_msgs
                            continue

                    # Check for queued messages
                    new_msgs = self._message_queue.copy()
                    self._message_queue.clear()
                    if new_msgs:
                        current_prompts = new_msgs
                        continue

                    # Enter idle — no timeout, no shield
                    self._status = RunStatus.idle
                    self._idle_event.clear()
                    await self._idle_event.wait()

                    # Check cancelled/closing BEFORE starting new Turn
                    if self._run_ctx.cancelled or self._closing:
                        break

                    self._status = RunStatus.running
                    new_msgs = self._message_queue.copy()
                    self._message_queue.clear()
                    if not new_msgs:
                        break
                    current_prompts = new_msgs
        finally:
            self._status = RunStatus.done
            with anyio.CancelScope(shield=True):
                self._run_ctx._run_handle.complete_event.set()

    async def steer(self, message: str) -> bool:
        """Inject message into active turn (asap) or wake idle RunHandle.

        Returns True if message was delivered, False if RunHandle is closing.
        M2: Checks _closing to avoid silently dropping messages.
        M3: Returns bool to match steer_callback type (Awaitable[bool]).
        R2: async def to match steer_callback: Callable[..., Awaitable[bool]].
        Use wrapper when assigning to steer_callback:
            run_ctx.steer_callback = lambda _sid, msg: run.steer(msg)
        """
        if self._closing:
            return False  # RunHandle is closing — reject steer
        if self._status == RunStatus.idle:
            self._message_queue.append(message)
            self._idle_event.set()
            return True
        # Running: try agent-type-specific real-time injection
        agent_run = self._run_ctx._run_handle.active_agent_run
        if agent_run is not None:
            agent_run.enqueue(message, priority="asap")
        else:
            self._message_queue.append(message)
        return True

    async def followup(self, message: str) -> bool:
        """Queue message for next Turn. Wake RunHandle if idle.

        Returns True if message was queued, False if RunHandle is closing.
        R2: async def to match steer_callback type.
        """
        if self._closing:
            return False
        self._message_queue.append(message)
        if self._status == RunStatus.idle:
            self._idle_event.set()
        return True

    def close(self) -> None:
        """Close the RunHandle. Wakes idle, signals closing flag.

        This is the primary lifecycle control. Called by:
        - async with __aexit__ (when caller exits the context)
        - close_session() (when session is being closed)
        Idempotent.
        """
        self._closing = True
        self._idle_event.set()  # Wake up if idle

    def cancel(self) -> None:
        """Cancel the RunHandle. Wakes idle, signals cancelled flag.

        Called by close_session() as fallback if close() doesn't
        terminate within timeout. Idempotent.
        """
        self._run_ctx.cancelled = True
        self._idle_event.set()  # Wake up if idle

    # Async context manager protocol
    async def __aenter__(self) -> RunHandle:
        return self

    async def __aexit__(self, *args: object) -> None:
        self.close()
```

#### Turn (abstract)

- **Responsibility**: Single reactive cycle. Agent-type-specific.
- **Interfaces**: `execute()` (async generator), `message_history` (property), `final_message` (property)

```python
class Turn(ABC):
    @abstractmethod
    def execute(self) -> AsyncGenerator[RichAgentStreamEvent, None]:
        """Execute one reactive cycle, yielding events.

        Implementations MUST be async generators (use `yield`).
        Lifecycle events (RunStartedEvent/StreamCompleteEvent) are published
        by RunHandle, not by execute(). Implementations yield only mid-stream
        events (PartDeltaEvent, ToolCallStartEvent, ToolCallCompleteEvent, etc.).
        """
        ...  # pragma: no cover — abstract
        yield  # type: ignore[unreachable]  # makes this an async generator

    @property
    @abstractmethod
    def message_history(self) -> list[ModelMessage]:
        """Updated message history after execution."""
        ...

    @property
    @abstractmethod
    def final_message(self) -> ChatMessage[Any]:
        """The final response message from this Turn."""
        ...
```

#### Event Mapping

The current `RunExecutor` contains ~170 lines of event mapping (run_executor.py L220-283) that convert pydantic-ai node-level events to `RichAgentStreamEvent` types. This logic is extracted into a shared `EventMapper` class used by both `NativeTurn` and (partially) by `ACPTurn`:

```python
class EventMapper:
    """Maps pydantic-ai stream events to RichAgentStreamEvent.

    Extracted from RunExecutor L220-283. Lives in orchestrator/event_mapper.py.
    Passthrough unmatched pydantic-ai events; matched events are replaced by
    RichAgentStreamEvent equivalents (M1).
    """

    def __init__(self, agent_name: str = "", message_id: str = "") -> None:
        # C6: Track pending tool calls by tool_call_id to match results
        # to their originating calls (replaces process_tool_event logic)
        self._pending_tool_calls: dict[str, str] = {}  # tool_call_id -> tool_name
        self._pending_tool_inputs: dict[str, dict[str, Any]] = {}  # tool_call_id -> args
        # R1: Needed for ToolCallCompleteEvent required fields
        self._agent_name = agent_name
        self._message_id = message_id

    def map_event(self, event: Any) -> RichAgentStreamEvent | None:
        """Map a pydantic-ai stream event to RichAgentStreamEvent.

        Returns None for events that should be skipped.
        C5: Constructs ToolCallStartEvent with all required fields
        (tool_call_id, tool_name, title, raw_input).
        """
        match event:
            case FunctionToolCallEvent(part=part) if hasattr(part, "tool_name"):
                # C5: Use correct field names and required fields
                tool_call_id = getattr(part, "tool_call_id", "")
                tool_name = part.tool_name
                tool_input = getattr(part, "args", {})
                self._pending_tool_calls[tool_call_id] = tool_name
                self._pending_tool_inputs[tool_call_id] = tool_input
                return ToolCallStartEvent(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    title=tool_name,
                    raw_input=tool_input,
                )
            case PartStartEvent(part=BaseToolCallPart()):
                tool_call_id = getattr(part, "tool_call_id", "")
                tool_name = getattr(part, "tool_name", "")
                if tool_call_id:
                    self._pending_tool_calls[tool_call_id] = tool_name
                return ToolCallStartEvent(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    title=tool_name,
                )
            case FunctionToolResultEvent(result=result):
                # C6: Access result attribute (not tool_name) to match
                # tool_call_id and construct ToolCallCompleteEvent
                # R1: Use correct field name tool_result (not result),
                # include all required fields (tool_input, agent_name, message_id)
                tool_call_id = getattr(result, "tool_call_id", "")
                tool_name = self._pending_tool_calls.pop(tool_call_id, "")
                tool_input = self._pending_tool_inputs.pop(tool_call_id, {})
                return ToolCallCompleteEvent(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_result=getattr(result, "content", ""),
                    agent_name=self._agent_name,
                    message_id=self._message_id,
                )
            case _:
                # M1: Passthrough raw pydantic-ai events for backward compat
                if isinstance(event, RichAgentStreamEvent):
                    return event
                return None
```

The `process_tool_event()` helper (run_executor.py helpers.py L30-75) that matches tool return parts to pending calls → `ToolCallCompleteEvent` is also extracted into `EventMapper.map_tool_result()`.

#### NativeTurn

- **Responsibility**: Thin wrapper over pydantic-ai `agentlet.iter()` → `next(node)` → `End`
- **Lines**: ~80 (including event mapping delegation, exception handling, terminal tool support)

```python
class NativeTurn(Turn):
    def __init__(
        self,
        agent: NativeAgent,
        prompts: str | list[str],
        run_ctx: AgentRunContext,
        message_history: list[ModelMessage],
    ):
        self._agent = agent
        self._prompts = prompts
        self._run_ctx = run_ctx
        self._message_history = message_history
        self._final_message: ChatMessage[Any] | None = None
        self._mapper = EventMapper()

    async def execute(self) -> AsyncGenerator[RichAgentStreamEvent, None]:
        agentlet = await self._agent.get_agentlet()
        # Build deps for tool access to AgentContext
        agent_deps = self._agent._build_deps(self._run_ctx)
        prompts = self._prompts if isinstance(self._prompts, list) else [self._prompts]

        async with agentlet.iter(
            prompts,  # positional: user_prompt content
            deps=agent_deps,
            message_history=self._message_history,
            usage_limits=self._agent._default_usage_limits,
        ) as agent_run:
            self._run_ctx._run_handle.active_agent_run = agent_run

            node = agent_run.next_node
            terminal_tool_completed = False
            while not isinstance(node, End):
                if self._run_ctx.cancelled:
                    break
                if isinstance(node, ModelRequestNode | CallToolsNode):
                    async with node.stream(agent_run.ctx) as stream:
                        async for event in stream:
                            mapped = self._mapper.map_event(event)
                            if mapped is not None:
                                yield mapped
                                # C6: Check for terminal tool completion
                                # via the mapped ToolCallCompleteEvent
                                # (not raw FunctionToolResultEvent)
                                if isinstance(mapped, ToolCallCompleteEvent):
                                    if self._agent._is_terminal_tool(mapped.tool_name):
                                        terminal_tool_completed = True
                            if terminal_tool_completed:
                                break
                if terminal_tool_completed:
                    break
                node = await agent_run.next(node)

            self._message_history = agent_run.all_messages()
            # Extract final response message from message history
            if self._message_history:
                last = self._message_history[-1]
                self._final_message = ChatMessage.from_model_message(last)
            self._run_ctx._run_handle.active_agent_run = None

    @property
    def message_history(self) -> list[ModelMessage]:
        return self._message_history

    @property
    def final_message(self) -> ChatMessage[Any]:
        if self._final_message is None:
            raise RuntimeError("final_message accessed before execute() completed")
        return self._final_message
```

**Implementation notes for helper methods referenced above:**

| Method | Source / Implementation |
|--------|------------------------|
| `self._agent._build_deps(run_ctx)` | Wraps existing `self._agent.get_context(input_provider=..., run_ctx=run_ctx)` — to be extracted as a method on `NativeAgent` during Phase 1 implementation |
| `self._agent._is_terminal_tool(tool_name)` | Delegates to `wolfharness.tools.base.is_terminal_tool()` — to be wrapped as a method on `NativeAgent` during Phase 1 |
| `ChatMessage.from_model_message(last)` | Constructs `ChatMessage` from a `ModelMessage` — to be implemented as a classmethod during Phase 1 (current code uses `ChatMessage.from_run_result()` at run_executor.py L330; this is a simplification) |
| `convert_acp_to_model_messages(acp_messages)` | Converts ACP message format to `list[ModelMessage]` — to be implemented in `agents/acp_agent/` during Phase 2 |
| `ACPTurn._map_acp_event(event)` | Maps ACP stream events to `RichAgentStreamEvent` — to be implemented in Phase 2, delegates to `EventMapper` where event types overlap |

#### ACPTurn

- **Responsibility**: Thin wrapper over ACP `session/prompt` → stream → complete
- **Lines**: ~30

```python
class ACPTurn(Turn):
    def __init__(
        self,
        acp_client: ACPClient,
        prompts: str | list[str],
        run_ctx: AgentRunContext,
        message_history: list[ModelMessage],
        session_id: str,
    ):
        self._client = acp_client
        self._prompts = prompts
        self._run_ctx = run_ctx
        self._message_history = message_history
        self._session_id = session_id
        self._final_message: ChatMessage[Any] | None = None

    async def execute(self) -> AsyncGenerator[RichAgentStreamEvent, None]:
        prompts = self._prompts if isinstance(self._prompts, list) else [self._prompts]
        response = await self._client.prompt(
            session_id=self._session_id,
            content=prompts[0] if len(prompts) == 1 else prompts,
        )
        async for event in self._client.stream_events(response):
            if self._run_ctx.cancelled:
                break
            mapped = self._map_acp_event(event)
            if mapped is not None:
                yield mapped
        # Convert ACP messages to ModelMessage format for consistency
        acp_messages = await self._client.get_messages(self._session_id)
        self._message_history = convert_acp_to_model_messages(acp_messages)
        if self._message_history:
            self._final_message = ChatMessage.from_model_message(self._message_history[-1])

    @property
    def message_history(self) -> list[ModelMessage]:
        return self._message_history

    @property
    def final_message(self) -> ChatMessage[Any]:
        if self._final_message is None:
            raise RuntimeError("final_message accessed before execute() completed")
        return self._final_message
```

### BaseAgent API

The `agent.run()` method returns a `RunHandle` object that is both an async context manager and an async iterator. This unifies v1 (single Turn) and v2 (persistent with idle) under a single API.

```python
class BaseAgent(ABC):
    @abstractmethod
    def create_turn(
        self,
        prompts: str | list[str],
        run_ctx: AgentRunContext,
        message_history: list[ModelMessage],
    ) -> Turn:
        """Create a single Turn. Agent-type-specific."""
        ...

    def run(
        self,
        prompt: str,
        *,
        run_ctx: AgentRunContext,
        message_history: list[ModelMessage],
        event_bus: EventBus,
        session: SessionState,
    ) -> RunHandle:
        """Create a RunHandle for this agent.

        Usage:
            # Single Turn (v1 compat):
            async with agent.run("prompt", ...) as run:
                async for event in run.start("prompt"):
                    ...

            # Persistent (v2):
            # start() called ONCE, steer() from a separate task
            async with agent.run("prompt", ...) as run:
                async for event in run.start("prompt"):
                    ...  # All Turns flow through this single async for
                # Between Turns, start() blocks on idle_event.wait()
                # A separate task calls run.steer("add tests") to wake
            # exit async with → run.close()

        NOTE: For v1 backward compatibility, BaseAgent.run_stream() wraps
        a single Turn as a plain async generator (no async with required).
        Existing run_stream() preamble (prompt conversion, ChatMessage
        construction, SessionPool Path A delegation) must wrap this call
        for full v1 compatibility. See Migration Plan Phase 1.
        """
        return RunHandle(self, run_ctx, event_bus, session)

    async def run_stream(
        self,
        prompt: str,
        *,
        run_ctx: AgentRunContext,
        message_history: list[ModelMessage],
        event_bus: EventBus,
        session: SessionState,
    ) -> AsyncGenerator[RichAgentStreamEvent, None]:
        """v1 compatible async generator. Single Turn, no idle.

        Wraps agent.run() for backward compatibility with code that
        uses `async for event in agent.run_stream(prompt):`.
        """
        async with self.run(
            prompt,
            run_ctx=run_ctx,
            message_history=message_history,
            event_bus=event_bus,
            session=session,
        ) as run:
            async for event in run.start(prompt):
                yield event
                # After StreamCompleteEvent, call close() to wake the Run
                # from idle and exit. Without this, start() blocks on
                # idle_event.wait() and the consumer deadlocks.
                if isinstance(event, StreamCompleteEvent):
                    run.close()
                    break
```

### Standalone Execution

Run/Turn architecture supports standalone execution without SessionPool. Idle/wait/steer are RunHandle primitives, not orchestrator infrastructure.

**Three execution modes:**

```python
# 1. Standalone single Turn (lightest weight, v1 compat)
async with NativeAgent(name="coder", model="...") as agent:
    async for event in agent.run_stream("write a function"):
        ...  # One Turn, exits at End

# 2. Standalone RunHandle with idle (no SessionPool)
# start() called ONCE, steer() from a separate task
async with NativeAgent(name="coder", model="...") as agent:
    event_bus = EventBus()  # Lightweight local EventBus
    run_ctx = AgentRunContext(...)
    session = SessionState(session_id="local")
    async with agent.run("write a function", run_ctx=run_ctx, ...) as run:
        # Consumer task: stays in async for across all Turns
        async for event in run.start("write a function"):
            ...  # Turn 1 events
            # After Turn 1, start() enters idle (blocks on idle_event.wait())
            # A separate task calls run.steer("now add tests") to wake
            # Turn 2 events continue through the same async for
    # exit async with → run.close()

# 3. Managed RunHandle (via SessionPool, multi-protocol subscription)
async with AgentPool("config.yml") as pool:
    run = await pool.receive_request(session_id, "write a function")
    async for event in pool.event_bus.subscribe(session_id):
        ...  # Any protocol server can also subscribe
```

**Why standalone works without SessionPool:**

- `RunHandle._message_queue` is a plain `list[str]` — no external dependency
- `asyncio.Event` for idle/wake is in-process — no distributed coordination needed
- `EventBus` can be created standalone (standalone mode creates a lightweight local EventBus)
- `turn_lock` comes from `SessionState`, which can be constructed independently

**Dependency layers:**

```
Standalone Turn          Standalone RunHandle       Managed RunHandle
      |                       |                           |
      v                       v                           v
   Turn.execute()         RunHandle.start()           RunHandle.start()
      |                       |                           |
      |                       +-- message_queue           +-- message_queue
      |                       +-- idle/wake               +-- idle/wake
      |                       +-- EventBus (local)        +-- EventBus (from SessionPool)
      |                                                   +-- session_id
      |                                                   +-- turn_lock (from SessionState)
      v                       v                           v
   agentlet.iter()        agentlet.iter()             agentlet.iter()
   (pydantic-ai)          (pydantic-ai)               (pydantic-ai)
```

### Multi-Protocol Subscription

The EventBus already supports multiple protocol servers subscribing to the same session. Run/Turn separation makes this cleaner by making RunHandle state visible:

```
Client A (ACP) --> ACP Server --+
                                 +--> EventBus[session:abc] --> RunHandle
Client B (OpenCode) --> OpenCode Server --+         ^
                                 |         |
Client C (AG-UI) --> AG-UI Server --+    Turn publishes events
```

Each `ProtocolEventConsumerMixin` instance independently subscribes to the EventBus and forwards events to its client. A 100-event replay buffer allows late-joining clients to catch up on history.

**RunHandle state visibility:**

```python
# Any protocol server can query RunHandle state
# NOTE: get_run() is new API to be added to SessionPool in Phase 1
run = session_pool.get_run(session_id)
run.status  # RunStatus.idle | running | done

# Any protocol server can subscribe
async for event in event_bus.subscribe(session_id, scope="session"):
    forward_to_client(event)

# Any protocol server can send messages
run.steer("new instruction from Client B")  # → message_queue → wake Turn
run.followup("process this after current Turn")
```

**Key difference from current architecture:** Currently `RunHandle` is per-request and not visible to protocol servers. With Run/Turn separation, `RunHandle` is a persistent, queryable object that all protocol servers can interact with.

**Limitations:**
1. EventBus is in-process — all protocol servers must be in the same AgentPool process
2. Turn execution is exclusive — `turn_lock` ensures only one Turn runs at a time per session
3. Concurrent steer from multiple clients — messages are FIFO queued, no priority conflict resolution

### Steer/Followup Unification

| Scenario | Current (4 branches) | Proposed (1-2 branches) |
|----------|---------------------|------------------------|
| Native + running | `agent_run.enqueue(priority="asap")` | `agent_run.enqueue(message, priority="asap")` (same) |
| Native + idle | `receive_request(priority="steer")` → new RunHandle | `message_queue.append()` + `idle_event.set()` |
| Non-native + running | `injection_manager.inject()` | `message_queue.append()` (queued for next Turn) |
| Non-native + idle | `_post_turn_injections` + `_trigger_auto_resume` | `message_queue.append()` + `idle_event.set()` (same as native idle) |

**Note on non-native steer during active turn**: The proposed design queues steer messages for the next Turn rather than injecting mid-run. This is a behavioral change for non-native agents — the current `injection_manager.inject()` provides tool-result augmentation (injecting context after tool calls). See "PromptInjectionManager Fate" below for how tool-result augmentation is preserved.

### PromptInjectionManager Fate

`PromptInjectionManager` has two distinct functions:

1. **Tool-result augmentation** (`inject()`/`consume()`): Injects context into the conversation after tool calls complete. This is a **per-Turn** concern and is **retained**. `NativeTurn` does not need it (pydantic-ai handles tool results natively), but `ACPTurn` and other non-native Turns use it within `execute()`.

2. **Follow-up prompt queuing** (`queue()`/`pop_queued()`): Defers messages to process after the current turn. This is **replaced** by `RunHandle._message_queue`. The RunHandle's idle/wake mechanism handles this natively.

**Resolution**: `PromptInjectionManager` is retained as a per-Turn utility for non-native agents. Its queuing methods (`queue()`/`pop_queued()`) are deprecated and removed in Phase 3. The `inject()`/`consume()` methods remain for tool-result augmentation within `ACPTurn.execute()`.

### SessionController Simplification

The restructured `RunHandle` absorbs run-level lifecycle logic from both `TurnRunner` and `SessionController`. `SessionController` is simplified to a **pure session registry** — it owns session CRUD, agent provisioning, and cross-session resource tracking, but no longer manages run creation, cancellation, or steer/followup routing.

**Architectural boundary**:

```
SessionController (session registry)     RunHandle (run lifecycle)
├── _sessions dict (session table)        ├── start() (idle/turn loop)
├── _session_agents dict (agent factory)  ├── steer() / followup()
├── _children (parent→child hierarchy)    ├── close() / cancel()
├── _session_scopes (CancelScope per session) ├── _message_queue
├── _max_concurrent_runs (global limit)   ├── _idle_event
├── MCP process tracking                  ├── RunStatus (idle/running/done)
├── Pending questions aggregation         └── complete_event
├── TTL cleanup loop
└── Storage persistence (checkpoint/resume)
```

**Methods absorbed by RunHandle (deleted from SessionController)**:

| Method | Current Location | Why RunHandle owns it |
|--------|-----------------|----------------------|
| `_create_run()` | `core.py:1533-1566` | RunHandle constructs itself; agent_type resolution inlines |
| `_cleanup_run()` | `core.py:1568-1578` | RunHandle already has `_cleanup_callback` + `complete_event` |
| `cancel_run_for_session()` | `core.py:1516-1531` | `RunHandle.cancel()` already exists; session→run lookup via `current_run_id` |

**Methods simplified in SessionController**:

| Method | Current | Simplified |
|--------|---------|------------|
| `receive_request()` | ~70 lines: session check + `_create_run` + task spawn + cleanup callback + busy-path steer/followup delegation | ~15 lines: session check + `max_concurrent_runs` check + delegate to `RunHandle.start()` (idle) or `RunHandle.steer()`/`.followup()` (busy) |
| `close_session()` | ~100 lines: mark closing + scope cancel + checkpoint + children + `turn_lock` wait + agent `__aexit__` + MCP decrement | ~60 lines: mark closing + scope cancel + checkpoint + children + `RunHandle.close()` + await `complete_event` + agent `__aexit__` + MCP decrement. The `turn_lock` wait + force-wake logic moves into RunHandle cleanup |

**Methods retained unchanged (27 methods)**:

Session registry CRUD (`get_or_create_session`, `get_session`, `list_sessions`, `find_sessions_by_agent_name`), agent factory (`get_or_create_session_agent`, `get_session_agent`), hierarchy (`get_children`, `get_parent`), storage (`_state_to_data`, `_should_checkpoint_on_close`, `_check_expired_calls`, `_save_close_checkpoint`, `_mark_session_closed`), background maintenance (`_cleanup_loop`, `_cleanup_expired_sessions`, `_start_cleanup_loop`, `start_cleanup_task`, `stop_cleanup_task`), MCP tracking (`_count_mcp_processes`, `_increment_mcp_count`, `_decrement_mcp_count`), pending questions (`list_pending_questions`, `cancel_all_pending_questions`, `cancel_session_pending_questions`, `list_pending_permissions`), internal session creation (`_get_or_create_session_locked`, `_close_session_unlocked`).

**Simplified `receive_request()` sketch**:

```python
async def receive_request(
    self, session_id: str, content: str | list[str],
    priority: str = "when_idle",
) -> None:
    session = self._sessions.get(session_id)
    if session is None:
        raise SessionNotFoundError(session_id)
    if session.closing:
        raise SessionClosingError(session_id)
    if self._max_concurrent_runs is not None:
        async with self._runs_lock:
            if len(self._runs) >= self._max_concurrent_runs:
                raise ConcurrencyLimitError(self._max_concurrent_runs)

    if session.current_run_id is None:
        # Idle: create RunHandle, register, start
        agent = await self.get_session_agent(session_id)
        run = RunHandle(agent, run_ctx, self._event_bus, session)
        self._runs[run.run_id] = run
        session.current_run_id = run.run_id
        task = asyncio.create_task(run.start(content))
        run._task = task
        task.add_done_callback(lambda t: self._runs.pop(run.run_id, None))
    else:
        # Busy: delegate to existing RunHandle
        run = self._runs.get(session.current_run_id)
        if run is None:
            # Race: run exited between check and lookup
            # Re-enter as new request
            session.current_run_id = None
            return await self.receive_request(session_id, content, priority)
        if priority in ("asap", "steer"):
            await run.steer(content if isinstance(content, str) else content[0])
        else:
            await run.followup(content if isinstance(content, str) else content[0])
```

### TurnRunner Fate

`TurnRunner` is **almost entirely absorbed** by the restructured `RunHandle`. Every method that contains run-level execution logic is replaced:

| TurnRunner Method | Lines | Replaced By | Fate |
|---|---|---|---|
| `run_loop()` | L2157-2204 | `RunHandle.start()` (outer while loop) | **Deleted Phase 3** |
| `_run_turn_unlocked()` | L1860-2132 | `RunHandle.start()` + `NativeTurn.execute()` / `ACPTurn.execute()` | **Deleted Phase 3** |
| `steer()` | L2300-2381 | `RunHandle.steer()` (unified, no branching) | **Deleted Phase 3** |
| `followup()` | L2383-2459 | `RunHandle.followup()` (unified, no branching) | **Deleted Phase 3** |
| `_process_queued_work()` | L2489-2559 | `RunHandle.start()` inner loop (idle→wake→next Turn) | **Deleted Phase 3** |
| `_trigger_auto_resume()` | L2561-2605 | `RunHandle` does not exit between turns | **Deleted Phase 3** |
| `_post_turn_injections` / `_post_turn_prompts` fields | L1789-1791 | `RunHandle._message_queue` | **Deleted Phase 3** |
| `_injection_locks` field | L1793 | Not needed (RunHandle owns single queue) | **Deleted Phase 3** |
| `_session_task_groups` field | L1795-1796 | `RunHandle` manages its own task group | **Deleted Phase 3** |
| `_runs` field | L1797 | Moves to `SessionController._runs` (already exists) | **Deleted Phase 3** |
| `_enable_auto_resume` / `_max_auto_resume` fields | L1799-1800 | No auto-resume concept (RunHandle never exits) | **Deleted Phase 3** |

**During Phase 1-2 (transition)**: `TurnRunner` retains its methods as **deprecated wrappers** that delegate to `RunHandle`:

```python
class TurnRunner:
    """Deprecated. Delegates to RunHandle. Will be removed in Phase 3."""
    
    def __init__(self, sessions: SessionController, event_bus: EventBus):
        self._sessions = sessions
        self._event_bus = event_bus
        import warnings
        warnings.warn(
            "TurnRunner is deprecated. Use RunHandle directly.",
            DeprecationWarning,
            stacklevel=2,
        )
    
    async def steer(self, session_id: str, message: str) -> bool:
        run = self._sessions._runs.get(self._sessions.get_session(session_id).current_run_id or "")
        if run:
            return await run.steer(message)
        return False
    
    async def followup(self, session_id: str, message: str) -> bool:
        run = self._sessions._runs.get(self._sessions.get_session(session_id).current_run_id or "")
        if run:
            return await run.followup(message)
        return False
```

**Phase 3**: `TurnRunner` class is **deleted entirely**. All references in protocol servers and `SessionPool` are replaced with direct `RunHandle` calls.

### Deprecated APIs and Breaking Changes

Key breaking changes are handled with `DeprecationWarning` rather than immediate fix, allowing gradual migration:

| API | Current | New | Deprecation Strategy |
|-----|---------|-----|---------------------|
| `TurnRunner.steer()` | 4-branch native/non-native | `RunHandle.steer()` unified | `DeprecationWarning` in Phase 1-2, deleted Phase 3 |
| `TurnRunner.followup()` | 4-branch native/non-native | `RunHandle.followup()` unified | `DeprecationWarning` in Phase 1-2, deleted Phase 3 |
| `TurnRunner.run_loop()` | Outer turn loop | `RunHandle.start()` | `DeprecationWarning` in Phase 1-2, deleted Phase 3 |
| `RunExecutor.execute()` | 448 lines | `NativeTurn.execute()` (~80 lines) + `EventMapper` (~180 lines) | `DeprecationWarning` in Phase 1, deleted Phase 3 |
| `RunExecutor` re-iteration loop | L389-414 | `RunHandle.start()` while loop | Deleted Phase 3 (no deprecation — internal) |
| `PromptInjectionManager.queue()` / `.pop_queued()` | Follow-up queuing | `RunHandle._message_queue` | `DeprecationWarning` in Phase 1-2, deleted Phase 3 |
| `SessionController._create_run()` | RunHandle factory | `RunHandle.__init__()` | Deleted Phase 3 (internal, no deprecation) |
| `SessionController._cleanup_run()` | Run cleanup callback | `RunHandle._cleanup_run()` | Deleted Phase 3 (internal, no deprecation) |
| `SessionController.cancel_run_for_session()` | Cancel by session ID | `RunHandle.cancel()` | `DeprecationWarning` in Phase 1-2, deleted Phase 3 |
| `SessionPool.inject_prompt()` / `.queue_prompt()` | Delegate to TurnRunner | Delegate to `RunHandle.steer()` / `.followup()` | `DeprecationWarning` in Phase 1-2, method bodies updated to delegate, old names kept as aliases |
| `RunHandle.complete_event` fires per-turn | Per-request | Per-RunHandle lifecycle (fires on close/cancel) | Documented as behavioral change; callers check `RunStatus` instead |

**Tests to update or delete**:

| Test File | Current Coverage | Action |
|-----------|-----------------|--------|
| `tests/test_turn_runner.py` (if exists) | TurnRunner.steer/followup/run_loop | **Phase 1-2**: Update to test `RunHandle.steer()`/`.followup()`/`.start()`. Keep TurnRunner tests as deprecated-path smoke tests. **Phase 3**: Delete TurnRunner tests. |
| `tests/test_run_executor.py` (if exists) | RunExecutor.execute() + re-iteration loop | **Phase 1**: Add `NativeTurn.execute()` tests. **Phase 3**: Delete RunExecutor tests. |
| `tests/orchestrator/test_session_controller.py` (if exists) | `receive_request`, `_create_run`, `_cleanup_run`, `cancel_run_for_session` | **Phase 1**: Update `receive_request` tests for delegation pattern. Delete `_create_run`/`_cleanup_run`/`cancel_run_for_session` tests (logic moves to RunHandle). Add `RunHandle` lifecycle tests. |
| `tests/test_prompt_injection.py` (if exists) | `PromptInjectionManager.queue()`/`.pop_queued()` | **Phase 1-2**: Mark queuing tests as `@pytest.mark.deprecated`. **Phase 3**: Delete queuing tests, keep `inject()`/`consume()` tests. |
| ACP integration tests | Non-native steer/followup via `injection_manager` | **Phase 2**: Update to test `RunHandle.steer()` for ACP path. Verify tool-result augmentation still works via `inject()`/`consume()`. |

### Components to Delete

| Component | File:Lines | Replaced By | Phase |
|-----------|-----------|-------------|-------|
| RunExecutor re-iteration loop | `run_executor.py:389-414` | `RunHandle.start()` while loop | 3 |
| `_post_turn_injections` / `_post_turn_prompts` fields | `core.py:1789-1791` | `RunHandle._message_queue` | 3 |
| `_injection_locks` field | `core.py:1793` | Not needed (single queue) | 3 |
| `_session_task_groups` field | `core.py:1795-1796` | `RunHandle` manages own task group | 3 |
| `_enable_auto_resume` / `_max_auto_resume` fields | `core.py:1799-1800` | No auto-resume (RunHandle never exits) | 3 |
| `TurnRunner._runs` field | `core.py:1797` | `SessionController._runs` (already exists) | 3 |
| `_trigger_auto_resume()` | `core.py:2561-2605` | `RunHandle` does not exit | 3 |
| `_process_queued_work()` | `core.py:2489-2559` | `RunHandle.start()` inner loop | 3 |
| `TurnRunner.steer()` | `core.py:2300-2381` | `RunHandle.steer()` (unified) | 3 |
| `TurnRunner.followup()` | `core.py:2383-2459` | `RunHandle.followup()` (unified) | 3 |
| `TurnRunner.run_loop()` | `core.py:2157-2204` | `RunHandle.start()` | 3 |
| `TurnRunner._run_turn_unlocked()` | `core.py:1860-2132` | `RunHandle.start()` + `Turn.execute()` | 3 |
| `TurnExecutor.execute()` task_group + cancel | `run_executor.py:373-438` | `NativeTurn.execute()` (simplified) | 3 |
| `RunExecutor.execute()` (entire) | `run_executor.py:1-448` | `NativeTurn.execute()` + `EventMapper` | 3 |
| `SessionController._create_run()` | `core.py:1533-1566` | `RunHandle.__init__()` | 3 |
| `SessionController._cleanup_run()` | `core.py:1568-1578` | `RunHandle._cleanup_run()` | 3 |
| `SessionController.cancel_run_for_session()` | `core.py:1516-1531` | `RunHandle.cancel()` | 3 |
| `PromptInjectionManager.queue()` / `.pop_queued()` | `prompt_injection.py:104-130` | `RunHandle._message_queue` | 3 |
| RunExecutor event mapping (L220-283) | `run_executor.py:220-283` | `EventMapper` class (extracted, shared) | 1 |
| `TurnRunner` class (entire) | `core.py:1751-2605` | `RunHandle` (absorbs all methods) | 3 |

### CancelScope Hierarchy

```
AgentPool.__aexit__
└─ SessionPool.shutdown()
   └─ Per-session CancelScope
      ├─ TaskGroup: Event consumers (active during idle)
      ├─ CancelScope: Run
      │  ├─ idle wait: await idle_event.wait() (NO shield, NO timeout)
      │  └─ Per-Turn: implicit (iter() context manager)
      └─ TaskGroup: Background tasks
```

**No `shield=True` on idle wait**: The idle `await self._idle_event.wait()` is NOT shielded. This allows `close_session()` to cancel the session's CancelScope (core.py `scope.cancel()`), which cascades to the idle wait, interrupting it immediately. The `close()` method (setting `_closing=True` + `idle_event.set()`) is the primary wake path; CancelScope cascade is the fallback.

**No timeout on idle wait**: Run waits indefinitely until woken. Callers who want timeout wrap with `anyio.move_on_after(N)`:

```python
# Caller-side timeout policy
async with anyio.move_on_after(300):
    async with agent.run("prompt", ...) as run:
        async for event in run.start("prompt"):
            ...
```

SessionPool may configure session-level idle policy:
```python
# SessionPool idle policy
if session.idle_duration > session.max_idle:
    await pool.close_session(session_id)  # Explicit close
```

**`close_session()` interaction**: The revised `close_session()` flow:
1. Call `RunHandle.close()` (sets `_closing=True`, wakes idle) — primary path
2. Set `session.closing = True`
3. Cancel session's CancelScope — cascades to RunHandle (interrupts idle if `close()` missed)
4. Acquire `turn_lock` (30s timeout) — RunHandle releases it on exit
5. Await `complete_event` (set in shielded scope at RunHandle exit) — 30s timeout
6. Timeout → force-cancel via `RunHandle.cancel()` + `cancel_run()`

### Subagent Interaction

The current `RunExecutor` re-iteration loop (L389-414) waits for `child_done_events` (subagent completions) and processes `queued_steer_messages`. The proposed design addresses this as follows:

**`child_done_events`**: These are `asyncio.Event` instances on `AgentRunContext` that signal subagent completion. In the new design, `RunHandle.start()` checks `child_done_events` **between Turns** (after a Turn completes, before entering idle). If any child events are set, the RunHandle processes the corresponding `queued_steer_messages` as the next Turn's prompts, mirroring the current re-iteration logic:

```python
# After Turn completes, before idle:
if self._run_ctx.child_done_events:
    # Wait for all pending subagents to complete
    for event in self._run_ctx.child_done_events:
        await event.wait()
    # Process queued steer messages from subagent completions
    steer_msgs = self._run_ctx.queued_steer_messages.copy()
    self._run_ctx.queued_steer_messages.clear()
    if steer_msgs:
        current_prompts = steer_msgs
        continue  # Start new Turn with subagent results
```

**`complete_background_task()`**: This method on `AgentRunContext` (context.py L136) signals subagent completion and calls `steer_callback`. In the new design, `steer_callback` is set by `RunHandle` to `RunHandle.steer()`, so subagent completions naturally route through the unified steer path.

**Key insight**: Subagent spawning creates child sessions with their own `RunHandle` instances. Each session has its own RunHandle. The parent RunHandle's `child_done_events` mechanism ensures it waits for child completions before entering idle. This is orthogonal to the Run/Turn separation — it's a between-Turns concern, not a within-Turn concern.

### Protocol Server Impact

Each protocol server (ACP, OpenCode, AG-UI, OpenAI API) calls into the orchestrator via `SessionPool` methods. The following table shows what changes at each call site:

| Call Site | Current Method | Proposed Method | Changes |
|-----------|----------------|-----------------|---------|
| `receive_request()` | `SessionController.receive_request()` → `TurnRunner.run_loop()` | `SessionController.receive_request()` → `RunHandle.start()` | RunHandle created in `receive_request()` (unchanged), but `run_loop()` replaced by `RunHandle.start()` |
| `steer()` | `TurnRunner.steer()` (4 branches) | `RunHandle.steer()` (unified) | `TurnRunner.steer()` becomes thin delegate |
| `followup()` | `TurnRunner.followup()` (4 branches) | `RunHandle.followup()` (unified) | `TurnRunner.followup()` becomes thin delegate |
| `close_session()` | `SessionController.close_session()` → wait `complete_event` | `SessionController.close_session()` → `RunHandle.close()` → wait `complete_event` | Must call `close()` before cancelling scope (see CancelScope Hierarchy) |
| `run_stream()` | `SessionPool.run_stream()` → `process_prompt()` → `TurnRunner.run_loop()` | `SessionPool.run_stream()` → `RunHandle.start()` | EventBus subscription unchanged; event source changes |

**Phase 1 routing**: During Phase 1 (native only), `SessionController.receive_request()` routes based on agent type:
- Native agent → create `RunHandle`, call `run_handle.start()`
- Non-native agent → existing `TurnRunner.run_loop()` path (unchanged)

This routing is implemented in `SessionController.receive_request()` with a single `isinstance(agent, NativeAgent)` check. The feature flag `AGENTPOOL_USE_RUN_TURN=true` gates whether native agents use the new path. When `false`, all agents use the existing `TurnRunner` path.

### Multi-Server Future Work

> **OQ#7 Resolved**: Multi-server is explicitly deferred to a follow-up RFC. The Run/Turn separation documented here is the necessary first step. Extensibility hooks are documented below to ensure the design does not preclude future distribution.

The Run/Turn separation is a **prerequisite** for multi-server support, but does not implement it. This RFC is explicitly in-process only.

**Current limitation**: All five key RunHandle components are in-process primitives:

| Component | Current Implementation | Multi-Server Would Need | Extensibility Hook |
|---|---|---|---|
| `RunHandle._message_queue` | In-process `list[str]` | Distributed queue (Redis Stream / NATS) | `_message_queue` is accessed only via `append()` and `copy()+clear()` — swap to any FIFO queue with `put()`/`drain()` |
| `asyncio.Event` (idle/wake) | Process-local | Distributed signal (Redis pub/sub) | `_idle_event` is accessed only via `set()`/`clear()`/`wait()` — swap to any async event with same interface |
| `asyncio.Lock` (turn_lock) | Process-local | Distributed lock (Redis SETNX) | `turn_lock` comes from `SessionState` — swap at construction time |
| `EventBus` | anyio memory object streams | Message broker (Redis pub/sub) | `EventBus` is already an abstraction — `publish()`/`subscribe()` interface unchanged |
| `RunHandle._status` (idle/running) | Memory variable | Shared state (Redis / DB) | `_status` is a simple enum — wrap in a property with sync backend |

**Why Run/Turn separation is the prerequisite**: The current architecture (RunHandle + TurnRunner + SessionController entangled together) cannot be distributed because state, execution, and queues are mixed in one object. Run/Turn separation cleanly divides:

- **RunHandle = state** → can be externalized to Redis/DB
- **Turn = execution** → can be leased to a worker process
- **steer/followup** → can publish to distributed queue
- **EventBus** → can publish to distributed stream

**Design decisions that preserve multi-server extensibility:**

1. **`_message_queue` and `_idle_event` are private fields, not hardcoded into `start()` control flow.** The `start()` method interacts with them via well-defined operations (`append`, `set`, `wait`, `clear`). A future `DistributedRunHandle` subclass can override these without rewriting `start()`.

2. **`Turn` is a separate object, not embedded in `RunHandle`.** This means Turn execution can be serialized and sent to a remote worker. `Turn.execute()` is a pure async generator with no back-references to `RunHandle` state (it receives `run_ctx` and `message_history` at construction).

3. **`EventBus` is injected, not created.** `RunHandle.__init__` receives `event_bus` as a parameter. A future distributed EventBus implementation can be injected without code changes to `RunHandle`.

4. **`steer()`/`followup()` are async methods.** This allows future implementations to `await` distributed queue operations without changing the call signature.

5. **`close()`/`cancel()` are sync but idempotent.** A future distributed implementation can wrap them in async adapters. The idempotent contract means retries are safe.

**Future architecture (out of scope for this RFC):**

```
Server A (ACP)     --> Redis --> RunHandle:abc (state)
Server B (OpenAI)  -->             status: running
                                   queue: [msg1, ...]
                                   turn_lock: held
                              --> Worker Process (owns agent config)
                                   Turn.execute()
```

This is a separate RFC. The Run/Turn separation documented here is the necessary first step.

---

## Security Considerations

### Threat Analysis

| Threat | Impact | Likelihood | Mitigation |
|--------|--------|------------|------------|
| Idle RunHandle holds `turn_lock` indefinitely | DoS — no new turns can start | Low | `close()` / `cancel()` always wakes; caller can wrap with `anyio.move_on_after(N)` for timeout policy |
| `message_queue` unbounded growth | Memory exhaustion | Low | Cap queue length; reject with `QueueFullError` if exceeded |
| `idle_event.set()` called from wrong task | Race condition on RunHandle state | Medium | `asyncio.Event` is task-safe; `set()`/`clear()` are synchronous |
| Cancel during idle doesn't wake RunHandle | Resource leak — RunHandle never exits | Medium | `cancel()` always calls `idle_event.set()` after setting `cancelled`; `close()` always calls `idle_event.set()` after setting `_closing` |
| `close()` not called (resource leak) | RunHandle holds turn_lock forever | Medium | `async with` ensures `close()` on context exit; SessionPool `close_session()` calls `close()` as fallback |

### Security Measures

- [ ] `message_queue` must have a configurable max length (default: 100)
- [ ] `cancel()` and `close()` must be idempotent and always wake the RunHandle
- [ ] `close_session()` must call `RunHandle.close()` before cancelling scope, with 30s timeout fallback to `cancel()`
- [ ] SessionPool may configure session-level idle policy (max idle duration before automatic `close_session()`)

### Hooks and Observability

**Hooks system**: The current hooks system (`pre_run`, `post_run`, `pre_tool_use`, `post_tool_use`) fires during `_run_turn_unlocked()`. In the new design:
- `pre_run` / `post_run` hooks fire **per-Turn** (inside `NativeTurn.execute()` / `ACPTurn.execute()`), not per-Run. This matches current semantics where hooks fire per turn.
- `pre_tool_use` / `post_tool_use` hooks fire within the pydantic-ai tool execution pipeline (unchanged for native agents). For non-native agents, hooks fire within `ACPTurn.execute()` when tool events are received.

**Storage and observability**: Each Turn is tracked as a separate interaction in storage, matching current behavior. `RunStartedEvent` and `StreamCompleteEvent` (published by `RunHandle`) carry the `run_id` that storage uses for interaction tracking. RunHandle idle periods are not tracked as interactions — only active Turns generate storage entries.

**EventBus lifecycle**: `RunHandle` uses the shared `EventBus` from `SessionController` (passed in constructor). No separate EventBus is created. The EventBus subscription lifecycle is managed by `ProtocolEventConsumerMixin` (unchanged). Event consumers remain active during idle, receiving `RunStartedEvent`/`StreamCompleteEvent` as Turns start/complete.

**`SessionState` interaction**: `RunStatus` (idle/running/done) is separate from `SessionState.current_run_id`. `current_run_id` is set when the RunHandle starts and cleared when the RunHandle exits (done/cancelled). During idle, `current_run_id` remains set (the RunHandle is alive), but `RunStatus.idle` indicates no active Turn. `close_session()` checks `RunStatus` to determine wake strategy.

---

## Implementation Plan

### Phases

#### Phase 1: Native Agent Run/Turn (v1 compatible)

- **Scope**: Implement `RunHandle` (restructured), `Turn` abstract, `NativeTurn`, `EventMapper`, `BaseAgent.run()` / `BaseAgent.run_stream()`. Simplify `SessionController.receive_request()`.
- **Key deliverables**:
  - `RunHandle` class restructured with idle/wake/close/cancel, `turn_lock` acquisition, `close_session()` interaction, `async with` protocol
  - Extensibility hooks documented for future multi-server support (see "Multi-Server Future Work")
  - `NativeTurn` wrapping pydantic-ai `iter()`/`next(node)` with exception handling, terminal tool support, event mapping delegation
  - `EventMapper` class extracted from `RunExecutor` L220-283 (shared utility)
  - `BaseAgent.run()` returning `RunHandle` (async context manager + async iterator)
  - `BaseAgent.run_stream()` as v1-compatible async generator wrapping single Turn
  - Unified `steer()`/`followup()` on `RunHandle`
  - **Simplify `SessionController.receive_request()`**: delegate to `RunHandle.start()` (idle) or `RunHandle.steer()`/`.followup()` (busy). Remove `_create_run()`, `_cleanup_run()`, `cancel_run_for_session()` from SessionController (move to RunHandle).
  - **`TurnRunner` deprecated**: Add `DeprecationWarning` to `TurnRunner.__init__()`. Methods become thin delegates to `RunHandle`.
- **Routing layer in `SessionController.receive_request()`**: route native agents to `RunHandle`, non-native to existing `TurnRunner` (gated by `AGENTPOOL_USE_RUN_TURN` feature flag)
- Migrate `close_session()` to handle `RunStatus.idle` and call `RunHandle.close()`
- Subagent interaction: `child_done_events` checked between Turns, `steer_callback` set to `RunHandle.steer()`
- **Tests**: Add `NativeTurn.execute()` tests, `RunHandle` lifecycle tests (idle/wake/steer/followup/close/cancel). Update `receive_request` tests for delegation pattern. Mark `TurnRunner` tests as `@pytest.mark.deprecated`.
- **Dependencies**: `introduce-anyio-structured-concurrency` (completed)
- **Rollback**: Keep `RunExecutor.execute()` as deprecated fallback; feature flag `AGENTPOOL_USE_RUN_TURN=true` (default: `false`)

#### Phase 2: Non-Native Agent (ACP) Migration

- **Scope**: Implement `ACPTurn`, migrate ACP path to `RunHandle`. Remove ACP-specific compensating complexity.
- **Deliverables**:
  - `ACPTurn` wrapping ACP `session/prompt` with `PromptInjectionManager` for tool-result augmentation
  - Remove `_post_turn_injections` / `_post_turn_prompts` (non-native) — `RunHandle._message_queue` replaces
  - Remove `_trigger_auto_resume()` (non-native) — `RunHandle` does not exit
  - Remove `_process_queued_work()` (non-native) — `RunHandle.start()` inner loop replaces
  - Remove `_run_turn_unlocked()` ACP branches — `ACPTurn.execute()` replaces
  - **`PromptInjectionManager.queue()`/`.pop_queued()` deprecated**: Add `DeprecationWarning`. Tool-result augmentation (`inject()`/`consume()`) retained.
  - **ACP integration tests**: Update to test `RunHandle.steer()` for ACP path. Verify tool-result augmentation still works.
- **Dependencies**: Phase 1 stable
- **Rollback**: `TurnRunner._run_turn_unlocked()` retained as deprecated path

#### Phase 3: Cleanup and Deprecation Removal

- **Scope**: Delete all compensating complexity, remove deprecated paths, delete `TurnRunner` class entirely.
- **Deliverables**:
  - **Delete `TurnRunner` class entirely** — all methods absorbed by `RunHandle`
  - Delete `RunExecutor` class entirely — replaced by `NativeTurn.execute()` + `EventMapper`
  - Delete `RunExecutor` re-iteration loop
  - Delete `SessionController._create_run()`, `_cleanup_run()`, `cancel_run_for_session()` (already moved to RunHandle in Phase 1)
  - Delete `PromptInjectionManager.queue()` / `.pop_queued()` (replaced by `RunHandle._message_queue`)
  - Delete `TurnRunner` fields: `_post_turn_injections`, `_post_turn_prompts`, `_injection_locks`, `_session_task_groups`, `_runs`, `_enable_auto_resume`, `_max_auto_resume`
  - Delete `RunExecutor` event mapping (already extracted to `EventMapper` in Phase 1)
  - Remove feature flag `AGENTPOOL_USE_RUN_TURN`
  - Update all `SessionPool` methods that delegate to `TurnRunner` to delegate to `RunHandle` directly
  - **Delete deprecated tests**: `TurnRunner` tests, `RunExecutor` tests, `PromptInjectionManager` queuing tests
  - **Update protocol server references**: Replace `TurnRunner` references with `RunHandle` in ACP/OpenCode/AG-UI/OpenAI API servers
- **Dependencies**: Phase 1 and Phase 2 stable for 1 release cycle
- **Validation**: Full test suite passes with `TurnRunner` and `RunExecutor` classes removed. No `DeprecationWarning` from orchestrator layer.

### Milestones

| Milestone | Description | Target | Status |
|-----------|-------------|--------|--------|
| M1: NativeTurn prototype | `NativeTurn.execute()` passes existing tests | Week 1 | Not Started |
| M2: RunHandle with idle | `agent.run()` supports idle/wake cycle via `async with` | Week 2 | Not Started |
| M3: Unified steer/followup | `RunHandle.steer()`/`.followup()` replace 4-branch implementation | Week 2 | Not Started |
| M4: SessionController simplified | `receive_request()` delegates to RunHandle; `_create_run`/`_cleanup_run`/`cancel_run_for_session` removed | Week 2 | Not Started |
| M5: TurnRunner deprecated | `TurnRunner` methods become thin delegates with `DeprecationWarning` | Week 2 | Not Started |
| M6: ACPTurn | `ACPTurn.execute()` passes ACP tests | Week 3 | Not Started |
| M7: Cleanup | Delete `TurnRunner` + `RunExecutor` classes, remove feature flag, delete deprecated tests | Week 4 | Not Started |

### Rollback Strategy

- Phase 1 uses feature flag `AGENTPOOL_USE_RUN_TURN=true` (default: `false`)
- If issues arise, disable flag to revert to `RunExecutor.execute()` path
- Phase 2 uses `AGENTPOOL_USE_RUN_TURN_FOR_ACP=true` (default: `false`)
- Phase 3 only executes after both phases are stable for 1 release cycle
- **Phase 3 is irreversible**: `TurnRunner` and `RunExecutor` classes are deleted. Git tags mark pre-Phase-1 and pre-Phase-3 states for easy revert.
- **Deprecation warnings** in Phase 1-2 give callers visibility into upcoming removals. `warnings.filterwarnings("error", category=DeprecationWarning)` can be used in CI to catch deprecated API usage early.

---

## Open Questions

1. ~~**Should `Run` be a separate class or a method on `BaseAgent`?**~~
   - **Resolved**: The Run concept is implemented as a separate class (`RunHandle`, restructured). Provides clean state isolation (RunHandle owns idle_event, message_queue, turn_lock lifecycle) without mixing session state into agent instances.

2. ~~**Should `RunHandle` be renamed to `Run` or kept as alias?**~~
   - **Resolved**: Keep `RunHandle` as the class name. The existing class is **restructured** (not renamed) to absorb Run semantics: idle/running/done states, message queue, steer/followup routing, `async with` lifecycle. Class name preserved for API stability — existing callers (`close_session()`, `cancel_run()`, `SessionPool._runs`, protocol servers) require no import changes. The concept "Run" (session-level execution context) is implemented by the `RunHandle` class.

3. ~~**What happens to `PromptInjectionManager` for non-native agents?**~~
   - **Resolved**: `PromptInjectionManager` is retained as a per-Turn utility for non-native agents. Tool-result augmentation (`inject()`/`consume()`) stays. Follow-up queuing (`queue()`/`pop_queued()`) is deprecated and removed in Phase 3, replaced by `RunHandle._message_queue`. See "PromptInjectionManager Fate" section.

4. ~~**Should `idle_timeout` be configurable per-session or per-agent?**~~
   - **Resolved**: No `idle_timeout` parameter. Run waits indefinitely until woken by `close()` or `steer()`/`followup()`. Timeout is caller's policy via `anyio.move_on_after(N)` or SessionPool session-level idle policy. This avoids race conditions between timeout and steer, and separates mechanism (Run) from policy (caller).

5. ~~**How does subagent spawning interact with the Run?**~~
   - **Resolved**: Each session has its own `RunHandle`. Subagent spawning creates child sessions with their own RunHandles. The parent RunHandle checks `child_done_events` between Turns (after Turn completes, before idle) and processes `queued_steer_messages` as next Turn prompts. `steer_callback` on `AgentRunContext` is set to `RunHandle.steer()`. See "Subagent Interaction" section.

6. ~~**How does `RunHandle.start()` interact with `async for` when multiple Turns are needed?**~~
   - **Resolved**: `start()` is called **once** and the consumer stays in `async for` across all Turns. Between Turns, `start()` blocks on `idle_event.wait()` — the event loop is free to run other tasks. `steer()`/`followup()`/`close()` are called from **separate tasks** (protocol servers, background tasks). The consumer's `async for` continues yielding events from the next Turn when a steer wakes the RunHandle. Calling `start()` twice would deadlock on `turn_lock` — this is by design, not a supported pattern.

7. ~~**Should multi-server / distributed Run be a follow-up RFC?**~~
   - **Resolved**: Multi-server is explicitly deferred to a follow-up RFC. The Run/Turn separation documented here is the necessary first step. Extensibility hooks are documented in the "Multi-Server Future Work" section to ensure the design does not preclude future distribution:
     - `_message_queue` and `_idle_event` are private fields with well-defined access patterns (append/copy/clear, set/clear/wait) — swappable to distributed equivalents
     - `Turn` is a separate object with no back-references to `RunHandle` state — can be serialized for remote execution
     - `EventBus` is injected (not created) — distributed implementation can be injected without code changes
     - `steer()`/`followup()` are async — allow future distributed queue operations without signature changes
     - `close()`/`cancel()` are sync idempotent — safe for retry-based distributed coordination

---

## Decision Record

> Complete this section after RFC review is concluded.

### Decision

**Status**: DRAFT

**Date**:

**Approvers**:

### Decision Summary

### Key Discussion Points

### Conditions of Approval

### Dissenting Opinions

---

## References

### Related Documents

- [RFC-0029: Agent Reactivation via Pending Prompt Queue](../draft/RFC-0029-agent-reactivation-pending-prompt-queue.md)
- [RFC-0037: Unify Steer and Followup Message Injection](../draft/RFC-0037-unify-steer-followup.md)
- [RFC-0021: Agent Concurrent Execution Safety](../implemented/RFC-0021-agent-concurrent-execution-safety.md)
- [ACP v2 Prompt Lifecycle RFD](https://github.com/nicholasgriffintn/agent-client-protocol/blob/main/docs/rfds/v2/prompt.mdx)
- [ACP PR #1261: session/inject](https://github.com/nicholasgriffintn/agent-client-protocol/pull/1261)
- [OpenSpec: introduce-anyio-structured-concurrency](../../../openspec/changes/introduce-anyio-structured-concurrency/)

### External Resources

- [pydantic-ai AgentRun source](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai/run.py)
- [pydantic-ai PendingMessageDrainCapability](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai/capabilities/_pending_messages.py)
- [anyio Condition documentation](https://anyio.readthedocs.io/en/stable/synchronization.html#condition)

### Appendix

#### A. Current Architecture Line Counts

| Component | File | Approximate Lines |
|-----------|------|-------------------|
| `RunHandle` | `orchestrator/run.py` | 150 |
| `RunExecutor` | `orchestrator/run_executor.py` | 440 |
| `TurnRunner` | `orchestrator/core.py` L1751-2605 | 850 |
| `SessionController` | `orchestrator/core.py` L774-1700 | 900 |
| `PromptInjectionManager` | `agents/prompt_injection.py` | 143 |
| **Total orchestrator** | | **~2513** |

#### B. Proposed Architecture Line Counts

| Component | File | Approximate Lines | Notes |
|-----------|------|-------------------|-------|
| `RunHandle` (restructured) | `orchestrator/run.py` (refactored) | ~200 | Absorbs TurnRunner run-loop + steer/followup + auto-resume |
| `Turn` (abstract) | `orchestrator/turn.py` (new) | ~25 | |
| `NativeTurn` | `agents/native_agent/turn.py` (new) | ~80 | Replaces RunExecutor.execute() (440 lines) |
| `ACPTurn` | `agents/acp_agent/turn.py` (new) | ~30 | Replaces TurnRunner ACP branches |
| `EventMapper` (extracted) | `orchestrator/event_mapper.py` (new) | ~180 | Extracted from RunExecutor L220-283 |
| `SessionController` (simplified) | `orchestrator/core.py` | ~750 | -150 lines: removed `_create_run`, `_cleanup_run`, `cancel_run_for_session`, simplified `receive_request` and `close_session` |
| `TurnRunner` | — | ~0 | **Deleted entirely** in Phase 3 (850 lines removed) |
| `RunExecutor` | — | ~0 | **Deleted entirely** in Phase 3 (440 lines removed) |
| `PromptInjectionManager` (trimmed) | `agents/prompt_injection.py` | ~80 | -63 lines: removed `queue()`/`pop_queued()`/`flush_pending_to_queue()` |
| **Total orchestrator** | | **~1345** |

**Net reduction: ~1168 lines (~46%)**. The reduction comes from:
- `TurnRunner` deleted entirely: -850 lines
- `RunExecutor` deleted entirely: -440 lines
- `SessionController` simplified: -150 lines
- `PromptInjectionManager` trimmed: -63 lines
- New components added: +515 lines (RunHandle +200, Turn +25, NativeTurn +80, ACPTurn +30, EventMapper +180)
