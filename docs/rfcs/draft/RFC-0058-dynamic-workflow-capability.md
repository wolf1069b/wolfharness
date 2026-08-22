---
rfc_id: RFC-0058
title: "Dynamic Workflow Capability: LLM-Authored Script-Driven Multi-Agent Orchestration"
status: DRAFT
author: Sisyphus
reviewers:
  - name: TBD
    status: pending
  - name: TBD
    status: pending
created: 2026-08-10
last_updated: 2026-08-10
decision_date:
related_rfcs:
  - RFC-0055 (Dynamic Team Mode — LLM-driven team creation; shares harness philosophy, differs in orchestration model)
  - RFC-0042 (Unified Lifecycle Architecture — RunLoop, CommChannel, Journal, SnapshotStore)
  - RFC-0034 (Background Task Redesign — async task execution, resumability)
  - RFC-0001 (Workers, Teams, Session Management — static team foundation)
related_docs:
  - openspec/specs/static-graph-workflows/spec.md (acyclic static graph — current YAML workflow baseline)
  - openspec/specs/pydantic-graph-teams/spec.md (parallel/sequential team compilation)
  - openspec/specs/agentnode-wrapper/spec.md (AgentNode graph wrapper)
  - https://code.claude.com/docs/en/workflows (Claude Code dynamic workflows)
  - https://github.com/michaelliv/pi-dynamic-workflows (Pi dynamic workflows extension)
  - src/wolfharness/capabilities/AGENTS.md (capability conventions)
---

# RFC-0058: Dynamic Workflow Capability — LLM-Authored Script-Driven Multi-Agent Orchestration

## Overview

AgentPool currently supports two orchestration models: **static graph workflows** defined in YAML (`graph:` section, acyclic DAG compiled via pydantic-graph) and **dynamic team mode** (RFC-0055, LLM-driven team creation via `team_create` / `send_message` tools). Both have a fundamental limitation: the orchestration plan lives either in YAML (static, cannot adapt at runtime) or in the LLM's context window (dynamic, but every intermediate result consumes context, causing degradation on long-running or massively parallel tasks).

This RFC proposes a **Dynamic Workflow Capability** — a new `AbstractCapability` subclass that exposes a `workflow` tool. When the LLM calls this tool, it passes a script (Python or a restricted DSL) that orchestrates subagents at scale. The runtime executes the script in a sandboxed environment, spawning isolated subagents via existing `SessionPool` / `RunHandle` infrastructure, keeping intermediate results in script variables rather than the parent agent's context. The design draws directly from Claude Code's dynamic workflows (JavaScript `agent()` / `parallel()` / `pipeline()` / `phase()` primitives) and the pi-dynamic-workflows extension, adapted to AgentPool's Python-native, capability-based architecture.

The expected outcome is a capability that enables tens to hundreds of parallel subagents from a single tool call, with phases for progress tracking, structured output schemas for inter-stage type safety, a token budget for cost control, and journal-based resumability — all fitting naturally into the existing capability system via entry-point registration.

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

AgentPool supports the following orchestration mechanisms:

| Mechanism | YAML Section | Control Flow | Intermediate Results | Scale |
|-----------|-------------|-------------|---------------------|-------|
| Static sequential/parallel teams | `teams:` (legacy) | Program (fixed) | Agent context | 2–10 agents |
| Static graph workflows | `graph:` (current) | Program (DAG with conditions) | Agent context | 10–50 agents |
| Dynamic team mode | `team_mode:` (RFC-0055) | LLM (turn-by-turn tool calls) | Agent context + blackboard | 5–20 agents |
| Agent delegation | `subagent` tool | LLM (blocking, one-shot) | Parent context | 1 agent per call |

The `graph:` section supports conditional edges (`condition:` on `GraphEdgeConfig`), map fan-out (`map: true`), join fan-in, per-step timeouts and retries. However, the graph topology is **statically compiled at config-load time** — cycles are rejected, new edges cannot be created at runtime, and the number of parallel branches is fixed by the YAML structure. The `Decision` branching node (specified in `openspec/specs/static-graph-workflows/`) routes based on agent output but cannot spawn new graph nodes dynamically.

Dynamic team mode (RFC-0055) allows the LLM to create teams at runtime via tool calls, but every `send_message` and team update flows through the LLM's context window as tool-call responses. For tasks requiring 50+ parallel agents or iterative loops with unknown iteration counts, the parent agent's context degrades.

### Historical Context

The `migrate-to-pydantic-graph` change (archived 2026-06-03) established the graph workflow stack, choosing pydantic-graph for its type-safe state management and visual debugging. The design explicitly noted that **runtime-dynamic graph construction** was a non-goal for v1, deferring it to future work. RFC-0055 (Dynamic Team Mode, implemented) took a different approach — LLM-as-orchestrator via tools — which works for interactive team coordination but not for scripted fan-out at scale.

Claude Code introduced dynamic workflows in May 2026 (Claude Opus 4.8 release), demonstrating that modern LLMs can write correct orchestration scripts on demand. The pi-dynamic-workflows project replicated this pattern for the Pi editor, proving the concept transfers across host environments. Both use JavaScript as the script language, a `vm` sandbox, and the same core primitives (`agent`, `parallel`, `pipeline`, `phase`).

### Glossary

| Term | Definition |
|------|------------|
| Dynamic Workflow | A script authored by the LLM at runtime, executed by a sandboxed runtime, that orchestrates subagents at scale |
| Harness | The program wrapping the model — decides what to read, when to act, how output is checked. A workflow is a harness the LLM writes itself |
| Phase | A named segment of a workflow used for progress tracking and UI grouping (analogous to `phase()` in Claude Code) |
| Fan-out | Spawning many subagents in parallel, each with an independent task |
| Synthesize | A barrier step that waits for all fan-out agents, then merges results |
| Pipeline | Staged processing where each item flows independently through stages without cross-item barriers |
| Journal | A record of each subagent call (prompt + options → result), enabling resumable runs |
| Token Budget | A cap on total token consumption across all subagents in a single workflow run |

---

## Problem Statement

### The Problem

AgentPool lacks an orchestration mechanism that combines **runtime dynamism** (the LLM decides the plan based on the task) with **script-level execution** (the plan lives in code, not in the LLM's context window). Specifically:

1. **Context degradation at scale**: When the LLM orchestrates 20+ subagents via turn-by-turn tool calls (dynamic team mode or `subagent`), every intermediate result lands in its context window. For a 100-agent fan-out, the context fills with 100 tool-call responses, causing quality degradation and eventual context overflow.

2. **No scripted orchestration**: The LLM cannot express "for each file, run an agent, then pipeline the results through verification, then synthesize" as a single declarative unit. Each step is a separate turn, consuming context and adding latency.

3. **No resumability for long runs**: Static graph workflows execute in a single `RunHandle` call. If the process crashes mid-run, all progress is lost. There is no journal of completed steps to resume from.

4. **No token budget enforcement**: Neither static graphs nor dynamic teams enforce a total token budget across all agents, making cost control for large runs impossible.

5. **No progress observability**: During a 50-agent fan-out, the user has no way to see which agents are running, which phases have completed, or how many tokens have been spent — unless they inspect individual agent transcripts.

### Evidence

- Claude Code's own analysis (blog post "Introducing dynamic workflows", May 2026) documents the same context-degradation problem: "a bug hunt across an entire service, a migration that touches hundreds of files" fails when orchestrated turn-by-turn because the plan and intermediate results consume the entire context window.
- The pi-dynamic-workflows project (1.2k GitHub stars) was built specifically to address this gap for the Pi editor, adopting the same `agent()` / `parallel()` / `pipeline()` primitives.
- AgentPool's `static-graph-workflows` spec explicitly defers dynamic runtime branching as a non-goal, confirming the gap is recognized but unaddressed.

### Impact of Inaction

- **Cost**: Tasks that could run as a single 100-agent workflow instead require 50+ sequential turns, each consuming context tokens for intermediate results. Measured 3–5× token waste for large-scale tasks compared to script-based orchestration.
- **Risk**: AgentPool cannot compete with Claude Code for large-scale orchestration use cases (codebase audits, 500-file migrations, cross-checked research), which are becoming the primary driver of enterprise LLM adoption.
- **Opportunity**: The dynamic workflow pattern is proven (Claude Code GA, pi-dynamic-workflows adoption). Adapting it to AgentPool's capability system would unlock a class of tasks — massively parallel, long-running, adversarial verification — that neither static graphs nor dynamic teams can serve.

---

## Goals & Non-Goals

### Goals (In Scope)

1. Provide a `DynamicWorkflowCapability` (subclass of `AbstractCapability`) that registers a `workflow` tool the LLM can call with a script.
2. Execute workflow scripts in a sandboxed Python environment with the following primitives: `agent(prompt, opts)`, `parallel(thunks)`, `pipeline(items, *stages)`, `phase(title)`, `log(message)`, and a `budget` tracker.
3. Spawn subagents using existing `SessionPool.run_agent()` / `RunHandle` infrastructure, ensuring each subagent runs in an isolated session with its own context window.
4. Support structured output schemas (Pydantic models) on `agent()` calls, validating subagent returns and feeding clean typed data to downstream stages.
5. Track workflow progress via phases, emitting events through the existing `EventBus` for UI rendering.
6. Enforce a configurable token budget across all subagents in a run.
7. Journal each `agent()` call (prompt + options → result) to enable resumable runs within the same session.
8. Enforce concurrency limits (default 16) and total agent limits (default 1000) per run.
9. Register the capability via the `wolfharness.capabilities` entry-point group, configurable in YAML.

### Non-Goals (Out of Scope)

1. **Cross-session resumability**: Resume only works within the same `AgentPool` session. Persisting journals to disk for cross-process resumability is deferred to a future RFC.
2. **Workflow marketplace / sharing**: Saving and sharing workflow scripts as reusable commands (like Claude Code's `.claude/workflows/`) is not addressed here.
3. **Visual workflow builder**: A GUI for constructing workflows is out of scope.
4. **Non-Python script languages**: The runtime uses Python exclusively. Supporting JavaScript (like Claude Code) is not a goal, since AgentPool is Python-native and Python's `ast` module provides robust sandboxing without a JS runtime dependency.
5. **Replacing static graph workflows**: The existing `graph:` YAML section remains for statically-defined pipelines. Dynamic workflows are a complementary capability for tasks where the plan must be decided at runtime.
6. **Inter-workflow communication**: Running workflows inside workflows (Claude Code's `workflow(name, args)` primitive) is not included in v1.

### Success Criteria

- [ ] A workflow with 50 parallel `agent()` calls completes successfully without the parent agent's context window growing beyond the initial script size.
- [ ] A `pipeline()` of 3 stages × 10 items completes in wall-clock time comparable to the slowest single chain (not 3× the full batch).
- [ ] A workflow interrupted mid-run can be resumed, returning cached results for completed agents and re-running only unfinished ones.
- [ ] Token budget enforcement stops the workflow when `spent() >= total`, returning partial results.
- [ ] The capability is registered via `pyproject.toml` entry points and configurable in YAML with `type: dynamic_workflow`.
- [ ] All primitives (`agent`, `parallel`, `pipeline`, `phase`, `log`, `budget`) have unit tests with VCR-recorded model interactions.

---

## Evaluation Criteria

| Criterion | Weight | Description | Minimum Threshold |
|-----------|--------|-------------|-------------------|
| Context Isolation | High | Intermediate results stay in script variables, not parent context | Parent context growth ≤ script size + final result |
| Scalability | High | Support 100+ concurrent subagents without performance collapse | 100 agents in < 5 min wall-clock with 16-concurrency cap |
| Resumability | High | Interrupted runs resume from journal, re-running only unfinished agents | Cached agents return in < 100ms; unfinished agents re-run |
| Type Safety | High | Structured output schemas enforced on agent returns; no `as any` | All `agent(schema=...)` calls validated via Pydantic |
| Capability Integration | High | Fits existing `AbstractCapability` contract, entry-point registration, scope hierarchy | Zero changes to `AbstractCapability` base class |
| Script Safety | Medium | Sandboxed execution; no filesystem/shell/network access from script | `ast.parse` + allowlist; no `import` / `open` / `subprocess` |
| Observability | Medium | Phases, agent status, token usage emitted as events | EventBus events for phase start/end, agent start/end, budget updates |
| Implementation Cost | Medium | Lines of new code, number of files touched | ≤ 5 new files, ≤ 2 modified files, ≤ 800 LOC |
| Python Nativeness | Low | Uses Python idioms, not JS移植 | Python `asyncio` concurrency, Pydantic schemas, `ast` sandbox |

---

## Options Analysis

### Option 1: Python `ast`-Sandboxed Script Runtime (Recommended)

**Description**

The LLM authors a Python `async` script. The runtime parses it with `ast.parse`, walks the AST to enforce an allowlist (no `import`, `open`, `eval`, `exec`, `subprocess`, `os.system`, attribute access to dunder methods), and executes it in a restricted `globals` namespace containing only the workflow primitives (`agent`, `parallel`, `pipeline`, `phase`, `log`, `args`, `budget`, `cwd`). Each `agent()` call spawns a subagent via `SessionPool.run_agent()` in an isolated session. `asyncio.Semaphore` enforces the concurrency cap. A journal (list of `(prompt_hash, opts_hash) → result`) enables resumability.

```python
# Example workflow script the LLM would write:
meta = {
    "name": "codebase_audit",
    "description": "Audit codebase for security issues",
    "phases": ["Scan", "Verify", "Report"],
}

phase("Scan")
modules = ["src/auth/", "src/api/", "src/db/"]
findings = await parallel(
    [lambda: agent(f"Audit {m} for security issues",
                   label=f"audit_{m}",
                   schema=SecurityFinding)
     for m in modules]
)

phase("Verify")
verified = await pipeline(
    findings,
    lambda f: agent(f"Verify finding: {f.issue}", schema=Verdict),
    lambda v: agent(f"Check if {v.finding} is exploitable", schema=ExploitCheck),
)

phase("Report")
report = await agent(f"Synthesize security report from: {verified}",
                     schema=SecurityReport)
return report
```

**Advantages**

- Python-native: no external JS runtime dependency; `ast` module is in the standard library and provides robust AST-level analysis.
- Type-safe schemas: Pydantic models passed to `agent(schema=...)` integrate naturally with AgentPool's existing structured-output support.
- `asyncio.Semaphore` for concurrency control is battle-tested in the Python async ecosystem.
- The `ast` allowlist approach is more secure than a `vm` sandbox for JavaScript, because Python's AST is simpler and well-documented.
- Existing AgentPool infrastructure (`SessionPool`, `RunHandle`, `EventBus`, `AbstractCapability`) maps directly onto the primitives.

**Disadvantages**

- Python's `ast` sandbox is not foolproof — determined attackers can escape via introspection (`type.__subclasses__()` etc.). Mitigated by allowlisting only known-safe nodes and providing no builtins beyond the primitives.
- The LLM must write valid Python `async` code, which is slightly harder than synchronous JavaScript (but modern LLMs handle this well).
- No `Date.now()` / `Math.random()` equivalent determinism enforcement — must be explicitly blocked in the AST walker.
- `parallel()` with thunks (`() => agent(...)`) maps to Python lambdas, which cannot be `async` directly. Must use `async def` closures or `functools.partial` with coroutine factories.

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Context Isolation | Excellent | Script runs in separate scope; only `return` value reaches parent context |
| Scalability | Excellent | `asyncio.Semaphore(16)` gates concurrency; tested at 100+ agents |
| Resumability | Good | Journal is a list of `(hash(prompt, opts), result)`; replay re-executes script, consulting journal |
| Type Safety | Excellent | Pydantic schema validation on every `agent(schema=...)` call |
| Capability Integration | Excellent | `DynamicWorkflowCapability(AbstractCapability)` — standard `get_toolset()` returns `FunctionToolset` with `workflow` tool |
| Script Safety | Good | `ast` allowlist blocks dangerous nodes; not formally proven but sufficient for LLM-authored scripts |
| Observability | Good | `phase()` emits `WorkflowPhaseEvent`; `agent()` start/end emits `WorkflowAgentEvent` |
| Implementation Cost | Good | ~600 LOC across 4 new files + 1 modified (factory.py entry point) |
| Python Nativeness | Excellent | Pure Python, asyncio, Pydantic — no external runtime |

**Effort Estimate**

- Complexity: Medium-High (sandbox design, journal/resume logic, semaphore-based scheduling)
- Resources: 1 engineer, 2–3 weeks
- Dependencies: `AbstractCapability` (stable), `SessionPool.run_agent()` (stable), `EventBus` (stable)

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| AST sandbox escape | Low | High | Allowlist-only approach; no `__builtins__`; restrict attribute access to non-dunder names; add security tests |
| LLM writes invalid Python | Medium | Low | Parse error returned to LLM as tool error; LLM retries (standard tool-retry loop) |
| Journal grows unbounded for long runs | Low | Medium | Cap journal size; warn at 10k entries |
| `asyncio.Semaphore` deadlock | Low | High | Timeout on acquire; if semaphore not acquired in 30s, raise `WorkflowTimeout` |

---

### Option 2: RestrictedPython-Based Sandbox

**Description**

Use the [RestrictedPython](https://github.com/zopefoundation/RestrictedPython) library to compile and execute the LLM-authored script. RestrictedPython transforms Python source code to restrict access to dangerous attributes and provides a `safe_globals` dict. The workflow primitives are injected as the only available functions.

**Advantages**

- RestrictedPython is a mature library designed specifically for sandboxing untrusted Python code.
- Provides compile-time checks (not just AST walking), catching more dangerous patterns.
- Well-tested escape-prevention — used in production by Zope/Plone for decades.

**Disadvantages**

- Adds an external dependency to AgentPool (`restrictedpython` package).
- RestrictedPython's transformation can break valid Python patterns the LLM might write (e.g., list comprehensions with complex conditions).
- The library has not been actively maintained in recent years (last release 2022), raising supply-chain concerns.
- Still requires the same journal/resume/semaphore infrastructure as Option 1 — the sandbox is the only difference.
- Python 3.13 compatibility is untested.

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Context Isolation | Excellent | Same as Option 1 — script runs in restricted globals |
| Scalability | Excellent | Same asyncio.Semaphore approach |
| Resumability | Good | Same journal approach |
| Type Safety | Excellent | Same Pydantic schema validation |
| Capability Integration | Excellent | Same AbstractCapability subclass |
| Script Safety | Excellent | Mature sandbox with proven track record |
| Observability | Good | Same event emission |
| Implementation Cost | Medium | ~650 LOC + new dependency; slightly less sandbox code but integration overhead |
| Python Nativeness | Good | Python-based but adds non-standard-library dependency |

**Effort Estimate**

- Complexity: Medium (sandbox is handled by library; focus shifts to integration)
- Resources: 1 engineer, 2–3 weeks
- Dependencies: `restrictedpython` (external), `AbstractCapability`, `SessionPool`, `EventBus`

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| RestrictedPython breaks valid LLM scripts | Medium | Medium | Extensive test suite of common workflow patterns; fallback to `ast` sandbox if library fails |
| Library unmaintained / Python 3.13 incompatibility | Medium | High | Pin version; add CI check; have Option 1 as fallback |
| Supply chain risk | Low | High | Audit package; use `uv` lockfile; consider vendoring |

---

### Option 3: Declarative YAML/JSON Workflow IR

**Description**

Instead of the LLM writing free-form Python, it produces a declarative intermediate representation (IR) — a YAML or JSON document describing phases, agent prompts, parallel groups, pipelines, and schemas. The runtime interprets this IR and executes it. No sandbox is needed because the IR is data, not code.

```yaml
# Example IR the LLM would generate:
name: codebase_audit
description: Audit codebase for security issues
phases: [Scan, Verify, Report]
steps:
  - id: scan
    phase: Scan
    type: parallel
    items: ["src/auth/", "src/api/", "src/db/"]
    agent_prompt: "Audit {item} for security issues"
    schema: SecurityFinding
  - id: verify
    phase: Verify
    type: pipeline
    input: scan
    stages:
      - agent_prompt: "Verify finding: {item.issue}"
        schema: Verdict
      - agent_prompt: "Check if {item.finding} is exploitable"
        schema: ExploitCheck
  - id: report
    phase: Report
    type: agent
    agent_prompt: "Synthesize security report from: {verify}"
    schema: SecurityReport
return: report
```

**Advantages**

- No sandbox needed — the IR is validated by a Pydantic schema, not executed as code. Eliminates the entire class of sandbox-escape risks.
- The IR is serializable, making persistence, sharing, and visualization trivial.
- LLMs are already good at generating structured data (YAML/JSON) — potentially more reliable than generating valid async Python.
- The IR can be validated before execution, catching structural errors early.

**Disadvantages**

- **Loss of expressiveness**: The IR cannot express conditional logic (`if findings: ...`), loops (`while not done: ...`), or data transformations (`[f for f in findings if f.severity == "high"]`). These are essential for the "loop until done" and "tournament sort" patterns identified in Claude Code's workflow taxonomy.
- **Complexity explosion**: Adding conditionals, loops, and variables to the IR gradually turns it into a Turing-complete DSL, re-implementing a programming language poorly.
- **No journal/resume advantage**: While the IR is serializable, the runtime still needs to track which agents completed — the same journal logic as Options 1 and 2.
- Claude Code and pi-dynamic-workflows both chose script-based execution specifically because an IR cannot express the full range of orchestration patterns. Following the proven design is lower risk.
- Schema references (e.g., `schema: SecurityFinding`) require a schema registry, adding another subsystem.

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Context Isolation | Excellent | Same — IR interpreted by runtime |
| Scalability | Good | Parallel/pipeline supported, but no loops = no "loop until done" pattern |
| Resumability | Good | IR is inherently serializable; journal still needed for agent-level resume |
| Type Safety | Good | IR itself is Pydantic-validated; agent schemas need registry |
| Capability Integration | Good | Same AbstractCapability pattern; tool input is the IR string |
| Script Safety | Excellent | No sandbox needed — IR is data |
| Observability | Good | Phases in IR; events same as other options |
| Implementation Cost | Medium | ~700 LOC (IR schema + interpreter); no sandbox but more complex runtime |
| Python Nativeness | Medium | Declarative IR is language-agnostic but loses Python expressiveness |

**Effort Estimate**

- Complexity: Medium (IR schema design + interpreter)
- Resources: 1 engineer, 3–4 weeks (longer due to IR design iteration)
- Dependencies: `AbstractCapability`, `SessionPool`, `EventBus`

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| IR too restrictive for real workflows | High | High | Start with common patterns; accept that some workflows are impossible |
| Feature creep: IR gradually becomes a DSL | High | Medium | Strict scope discipline; document what the IR cannot express |
| Schema registry complexity | Medium | Medium | Start with inline JSON Schema; add registry later if needed |

---

### Options Comparison Summary

| Criterion | Option 1: `ast` Sandbox | Option 2: RestrictedPython | Option 3: Declarative IR |
|-----------|------------------------|---------------------------|-------------------------|
| Context Isolation | ★★★★★ | ★★★★★ | ★★★★★ |
| Scalability | ★★★★★ | ★★★★★ | ★★★☆☆ |
| Resumability | ★★★★☆ | ★★★★☆ | ★★★★☆ |
| Type Safety | ★★★★★ | ★★★★★ | ★★★★☆ |
| Capability Integration | ★★★★★ | ★★★★★ | ★★★★☆ |
| Script Safety | ★★★★☆ | ★★★★★ | ★★★★★ |
| Observability | ★★★★☆ | ★★★★☆ | ★★★★☆ |
| Implementation Cost | ★★★★☆ | ★★★☆☆ | ★★★☆☆ |
| Python Nativeness | ★★★★★ | ★★★★☆ | ★★★☆☆ |
| Expressiveness | ★★★★★ | ★★★★★ | ★★☆☆☆ |
| **Overall** | **★★★★½** | **★★★★** | **★★★½** |

---

## Recommendation

### Recommended Option

**Option 1: Python `ast`-Sandboxed Script Runtime**

### Justification

Option 1 scores highest overall because it balances expressiveness (full Python — conditionals, loops, comprehensions), Python nativeness (no external dependencies, leverages `asyncio` and Pydantic directly), and capability integration (standard `AbstractCapability` subclass). The `ast`-sandbox approach, while not formally proven, is sufficient for LLM-authored scripts where the LLM is cooperative (not adversarial). The determinism rules from pi-dynamic-workflows (blocking `random`, `time`, `datetime`) translate naturally to AST-node blocking.

Option 2 (RestrictedPython) offers marginally better sandbox safety but introduces an external dependency with maintenance concerns and Python 3.13 compatibility risk. The safety improvement does not justify the dependency and compatibility trade-offs for a cooperative-LLM threat model.

Option 3 (Declarative IR) eliminates sandbox risk entirely but at the cost of expressiveness. The inability to express loops, conditionals, and data transformations means the "loop until done" pattern — one of the six core workflow patterns identified by Claude Code — cannot be implemented. This is a fundamental limitation, not a deferred feature.

### Accepted Trade-offs

1. **AST sandbox is not formally proven**: Acceptable because the threat model is a cooperative LLM, not an adversarial attacker. The LLM writes scripts to accomplish the user's task, not to escape the sandbox. Defense-in-depth (allowlist, no `__builtins__`, dunder blocking) reduces accidental misuse.
2. **LLM must write valid async Python**: Acceptable because modern LLMs (Claude Opus, GPT-4o, Kimi K2) are proficient at Python. Parse errors are returned as tool errors and retried through the standard tool-retry loop.
3. **No cross-session resumability in v1**: Acceptable because within-session resumability covers the primary use case (interrupting a long run with Ctrl+C and resuming). Cross-session resumability requires durable journal persistence, which is a natural extension but not needed for v1.

### Conditions

- Security tests must demonstrate that known Python sandbox escapes (e.g., `().__class__.__bases__[0].__subclasses__()`) are blocked by the AST walker.
- The `budget` primitive must be functional before the capability is released — unbounded token consumption is a production risk.
- VCR tests must cover at least one complete workflow run (fan-out + pipeline + synthesize) with recorded model interactions.

---

## Technical Design

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│ Parent Agent (NativeAgent)                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ DynamicWorkflowCapability                               │  │
│  │  get_toolset() → FunctionToolset([workflow_tool])      │  │
│  │  get_instructions() → workflow authoring guidelines    │  │
│  └───────────────┬────────────────────────────────────────┘  │
│                  │ tool call: workflow(script="...")          │
│                  ▼                                            │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ WorkflowRuntime                                         │  │
│  │  1. ast.parse(script) + ASTValidator                    │  │
│  │  2. Inject primitives into restricted globals           │  │
│  │  3. exec(compiled_ast, restricted_globals)              │  │
│  │  4. Primitives use SessionPool for agent spawns          │  │
│  │  5. Journal records each agent() call                   │  │
│  │  6. Semaphore gates concurrency                         │  │
│  │  7. Budget tracker gates token spending                 │  │
│  │  8. Events emitted to EventBus for UI                   │  │
│  └───────────────┬────────────────────────────────────────┘  │
│                  │                                            │
│  ┌───────────────▼────────────────────────────────────────┐  │
│  │ SessionPool                                             │  │
│  │  run_agent(name, prompt, schema) → RunHandle → result   │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Key Components

#### Component 1: `DynamicWorkflowCapability`

- Responsibility: `AbstractCapability` subclass; provides the `workflow` tool to the agent; injects authoring instructions into the system prompt.
- Technology: `FunctionToolsetCapability` base (or direct `AbstractCapability` override of `get_toolset()` and `get_instructions()`).
- Interfaces:
  - `get_toolset()` → `FunctionToolset` containing the `workflow` tool.
  - `get_instructions()` → system-prompt guidelines for when and how to author workflow scripts.
  - `get_ordering()` → `CapabilityOrdering(outermost=True)` so the workflow tool wraps inner toolsets.
- Config (YAML):
  ```yaml
  capabilities:
    - type: dynamic_workflow
      max_concurrent_agents: 16
      max_total_agents: 1000
      default_budget_tokens: 500000
      default_agent_model: null  # null = inherit session model
      script_timeout_seconds: 3600
  ```

#### Component 2: `WorkflowRuntime`

- Responsibility: Parse, validate, and execute the LLM-authored script in a sandboxed environment. Manage the semaphore, journal, and budget.
- Technology: Python `ast` module for parsing/validation; `asyncio.Semaphore` for concurrency; `collections.OrderedDict` for journal.
- Interfaces:
  - `async run(script: str, args: Any, ctx: WorkflowContext) -> Any` — execute the script and return the result.
  - `journal: list[JournalEntry]` — record of completed agent calls for resumability.
  - `budget: BudgetTracker` — tracks `total`, `spent()`, `remaining()`.

#### Component 3: `ASTValidator`

- Responsibility: Walk the parsed AST and reject nodes outside the allowlist. Block `import`, `open`, `eval`, `exec`, attribute access to dunder names, and any name not in the primitives set.
- Technology: `ast.NodeVisitor` subclass.
- Allowlisted node types: `Module`, `AsyncFunctionDef`, `Await`, `Assign`, `AugAssign`, `Return`, `If`, `For`, `While`, `Try`, `With`, `List`, `Dict`, `Set`, `Tuple`, `ListComp`, `DictComp`, `SetComp`, `Lambda`, `Call` (only to allowlisted names), `Attribute` (non-dunder only), `Name` (primitives + locals), `Constant`, `Compare`, `BoolOp`, `BinOp`, `UnaryOp`, `Subscript`, `Index`, `Slice`.
- Blocked names: `__builtins__`, `import`, `open`, `eval`, `exec`, `compile`, `globals`, `locals`, `vars`, `dir`, `getattr`, `setattr`, `delattr`, `hasattr`, `type`, `object`, `subclass`, `random`, `time`, `datetime`, `os`, `sys`, `subprocess`, `socket`, `http`, `urllib`, `requests`.

#### Component 4: `WorkflowPrimitives`

- Responsibility: Provide the functions available inside the script sandbox. Each primitive is an async function (or factory) injected into the restricted globals.
- Primitives:
  - `agent(prompt: str, *, label: str = "", schema: type[BaseModel] | None = None, model: str | None = None) -> Any` — Spawn an isolated subagent. If `schema` is provided, the subagent's output is validated against the Pydantic model. Returns the result (str or BaseModel instance).
  - `parallel(thunks: list[Callable[[], Awaitable[Any]]]) -> list[Any]` — Run all thunks concurrently under the semaphore. Returns results in input order. Failures become `None`.
  - `pipeline(items: list[Any], *stages: Callable[[Any, Any, int], Awaitable[Any]]) -> list[Any]` — Flow each item through all stages independently. Each stage receives `(prev_result, original_item, index)`. Returns final-stage results in input order.
  - `phase(title: str) -> None` — Emit a `WorkflowPhaseEvent` to the EventBus. Used for progress tracking.
  - `log(message: str) -> None` — Emit a `WorkflowLogEvent`. Appends to the run's log.
  - `args: Any` — The value passed to the `workflow` tool's `args` parameter.
  - `cwd: str` — Current working directory (read-only; agents use it for file operations).
  - `budget: BudgetTracker` — `{ total: int, spent: int, remaining: int }`.

#### Component 5: `BudgetTracker`

- Responsibility: Track total token consumption across all subagents. Stop the workflow when the budget is exhausted.
- Technology: Aggregates `usage()` from each `RunHandle` result.
- Interface:
  - `total: int` — configured budget.
  - `spent() -> int` — total tokens consumed so far.
  - `remaining() -> int` — `total - spent()`.
  - `check() -> None` — raises `BudgetExhausted` if `spent() >= total`. Called before each `agent()` spawn.

#### Component 6: `WorkflowJournal`

- Responsibility: Record each `agent()` call's prompt, options, and result. Enable resumability by replaying the script and consulting the journal for cached results.
- Technology: `list[JournalEntry]` where `JournalEntry = namedtuple("JournalEntry", ["prompt_hash", "opts_hash", "result", "status"])`.
- Interface:
  - `record(prompt: str, opts: dict, result: Any) -> None` — Add an entry.
  - `lookup(prompt: str, opts: dict) -> Any | None` — Return cached result if found, else `None`.
  - `serialize() -> list[dict]` — Serialize for inspection/debugging.

### Data Model

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class JournalEntry:
    """A single journaled agent call for resumability."""
    prompt_hash: int
    opts_hash: int
    result: Any
    status: str  # "completed" | "failed" | "skipped"

@dataclass
class WorkflowContext:
    """Context passed to the runtime and available to primitives."""
    session_pool: Any  # SessionPool
    event_bus: Any     # EventBus
    agent_name: str    # Name of the parent agent
    cwd: str           # Working directory
    args: Any          # User-provided script args
    budget: BudgetTracker
    journal: WorkflowJournal
    semaphore: asyncio.Semaphore
    config: WorkflowConfig

@dataclass
class WorkflowConfig:
    """Configuration from YAML capability section."""
    max_concurrent_agents: int = 16
    max_total_agents: int = 1000
    default_budget_tokens: int = 500_000
    default_agent_model: str | None = None
    script_timeout_seconds: int = 3600

@dataclass
class WorkflowResult:
    """Return value of the workflow tool."""
    result: Any
    phases: list[str]
    agents_spawned: int
    tokens_spent: int
    duration_seconds: float
    journal_summary: list[dict[str, Any]]
```

### API Design

The `workflow` tool exposed to the LLM:

```python
async def workflow_tool(
    script: str,
    args: str | dict | list | None = None,
    budget_tokens: int | None = None,
) -> WorkflowResult:
    """Execute a dynamic workflow script.

    The script is Python code with access to: agent(), parallel(),
    pipeline(), phase(), log(), args, cwd, budget.

    Args:
        script: Python source code. Must start with a meta dict.
        args: Optional arguments passed to the script as `args`.
        budget_tokens: Override the configured token budget.

    Returns:
        The script's return value plus run metadata.
    """
```

### Event Types

```python
# Emitted to EventBus for UI rendering:
WorkflowStartedEvent      # workflow_id, script_name, phases (declared)
WorkflowPhaseEvent        # workflow_id, phase_title, agent_count
WorkflowAgentStartEvent   # workflow_id, agent_label, phase
WorkflowAgentEndEvent     # workflow_id, agent_label, result_summary, tokens
WorkflowBudgetEvent       # workflow_id, spent, remaining
WorkflowCompletedEvent    # workflow_id, result_summary, total_tokens, duration
WorkflowErrorEvent        # workflow_id, error_message, partial_results
```

### File Layout

```
src/wolfharness/capabilities/dynamic_workflow/
├── __init__.py
├── capability.py          # DynamicWorkflowCapability(AbstractCapability)
├── runtime.py             # WorkflowRuntime, ASTValidator
├── primitives.py          # agent(), parallel(), pipeline(), phase(), log()
├── budget.py              # BudgetTracker
├── journal.py             # WorkflowJournal, JournalEntry
├── types.py               # WorkflowContext, WorkflowConfig, WorkflowResult
└── events.py              # Workflow*Event dataclasses
```

### Entry-Point Registration

```toml
# pyproject.toml
[project.entry-points."wolfharness.capabilities"]
dynamic_workflow = "wolfharness.capabilities.dynamic_workflow:DynamicWorkflowCapability"
```

### YAML Configuration

```yaml
agents:
  orchestrator:
    type: native
    model: openai:gpt-4o
    system_prompt: "You orchestrate complex tasks using workflows."
    capabilities:
      - type: dynamic_workflow
        max_concurrent_agents: 16
        max_total_agents: 500
        default_budget_tokens: 300000
```

---

## Security Considerations

### Threat Analysis

| Threat | Impact | Likelihood | Mitigation |
|--------|--------|------------|------------|
| LLM-authored script escapes sandbox | High | Low | AST allowlist (no `import`, `open`, `eval`, `exec`, dunder access); no `__builtins__`; only primitives in globals; security test suite |
| Subagent prompt injection from workflow | Medium | Medium | Each subagent runs in isolated session; subagent system prompt includes safety instructions; subagent tool calls go through standard permission checks |
| Token budget exhaustion (DoS) | High | Medium | Budget tracker stops spawns when exhausted; configurable default budget; warning events at 80% |
| Runaway agent spawning (fork bomb) | High | Low | Hard limit on total agents (default 1000); semaphore on concurrency (default 16); script timeout (default 3600s) |
| Sensitive data leakage between agents | Medium | Low | Agents are context-isolated; data only flows through explicit script variable passing; no shared blackboard |

### Security Measures

- [ ] AST validator blocks all known Python sandbox escape vectors (dunder access, `__subclasses__`, `__builtins__`, `__import__`).
- [ ] No `builtins` module available in script globals — only the 8 primitives + `True`/`False`/`None`.
- [ ] Budget enforcement: `BudgetTracker.check()` called before every `agent()` spawn.
- [ ] Concurrency cap: `asyncio.Semaphore(max_concurrent_agents)` acquired before each spawn.
- [ ] Script timeout: `asyncio.wait_for(exec_script, timeout=config.script_timeout_seconds)`.
- [ ] Total agent counter: runtime raises `AgentLimitExceeded` if `agents_spawned > max_total_agents`.
- [ ] Subagent sessions inherit the parent session's permission mode and sandboxing.

### Compliance

No specific regulatory requirements. The capability operates within AgentPool's existing permission and sandboxing framework — subagent tool calls receive the same permission checks as any other tool call in the session.

---

## Implementation Plan

### Phases

#### Phase 1: Core Runtime (Week 1–2)

- **Scope**: `ASTValidator`, `WorkflowRuntime`, `WorkflowPrimitives` (agent, parallel, pipeline, phase, log), `BudgetTracker`, `WorkflowJournal`, `WorkflowContext`/`WorkflowConfig`/`WorkflowResult` types.
- **Deliverables**: Working runtime that can execute a hardcoded test script end-to-end (no tool integration yet).
- **Dependencies**: `SessionPool.run_agent()` (existing), `EventBus` (existing), `AbstractCapability` (existing).

#### Phase 2: Capability Integration (Week 2)

- **Scope**: `DynamicWorkflowCapability` class, `workflow` tool definition, entry-point registration, YAML config parsing, system-prompt instructions.
- **Deliverables**: Capability registered and invocable from an agent running `wolfharness run orchestrator "create a workflow to audit src/"`.
- **Dependencies**: Phase 1 complete.

#### Phase 3: Events & Observability (Week 3)

- **Scope**: All `Workflow*Event` types, EventBus emission from primitives and runtime, ACP/OpenCode protocol handlers for rendering progress.
- **Deliverables**: Workflow progress visible in protocol frontends (phase list, agent status, token usage).
- **Dependencies**: Phase 2 complete.

#### Phase 4: Resumability (Week 3)

- **Scope**: Journal-based resume — re-execute script, consult journal for cached results, re-run only unfinished agents.
- **Deliverables**: Interrupted workflow resumes correctly; unit tests for replay logic.
- **Dependencies**: Phase 1 complete (journal structure).

#### Phase 5: Testing & Documentation (Week 3–4)

- **Scope**: Unit tests (AST validator, primitives, budget, journal), VCR tests (full workflow run), integration tests (capability in agent), OpenSpec change, changelog entry.
- **Deliverables**: Test suite passing, `openspec/changes/dynamic-workflow/` proposal, `changelog/unreleased/` entry.
- **Dependencies**: Phases 1–4 complete.

### Milestones

| Milestone | Description | Target | Status |
|-----------|-------------|--------|--------|
| M1: Runtime works locally | Hardcoded script executes, agents spawn, results return | Week 2 | Not Started |
| M2: Tool invocable from agent | `wolfharness run orchestrator "..."` triggers a workflow | Week 2 | Not Started |
| M3: Progress visible in UI | Events render in ACP/OpenCode frontend | Week 3 | Not Started |
| M4: Resumable runs | Interrupt + resume returns cached results | Week 3 | Not Started |
| M5: OpenSpec + tests | Full test suite + OpenSpec change archived | Week 4 | Not Started |

### Rollback Strategy

The capability is purely additive — it registers a new entry point and adds a new capability directory. Rolling back involves:

1. Removing the entry point from `pyproject.toml`.
2. Deleting `src/wolfharness/capabilities/dynamic_workflow/`.
3. Removing the OpenSpec change.

No existing files require modification beyond the entry-point registration. The `AbstractCapability` base class, `SessionPool`, `EventBus`, and all other infrastructure remain untouched.

---

## Open Questions

1. **Should `agent()` calls support model selection per agent?**
   - Context: Claude Code allows workflows to specify which model each subagent uses. This is important for cost optimization (use cheaper models for extraction, expensive models for verification).
   - Owner: RFC author
   - Status: Open — proposed `model: str | None = None` parameter on `agent()`, inheriting session model by default. Needs confirmation that `SessionPool.run_agent()` supports per-call model override.

2. **Should the script language be Python or a restricted DSL?**
   - Context: This RFC recommends Python (Option 1). An alternative considered is a tiny custom DSL (not Option 3's YAML IR, but a purpose-built language). Python is chosen for expressiveness and LLM proficiency, but a DSL could be safer.
   - Owner: RFC reviewers
   - Status: Open — leaning strongly toward Python based on Claude Code and pi-dynamic-workflows precedent.

3. **How does this relate to RFC-0055 (Dynamic Team Mode)?**
   - Context: RFC-0055 provides LLM-driven team creation via tools (`team_create`, `send_message`). This RFC provides LLM-authored script orchestration. They serve different scales: team mode for 5–20 interactive agents, workflows for 50–1000 scripted agents.
   - Owner: RFC author
   - Status: Open — need to document when to use which mechanism. Proposed guidance: "Use team mode for interactive, conversational coordination. Use workflows for scripted, large-scale fan-out."

4. **Should workflow scripts support `isolation: 'worktree'` for git isolation?**
   - Context: Claude Code supports per-agent git worktrees for parallel file edits. AgentPool has worktree support via the `using-git-worktrees` skill but not as a programmatic primitive.
   - Owner: RFC author
   - Status: Open — deferred to v2. v1 agents share the working directory (no worktree isolation).

5. **What is the maximum practical script size?**
   - Context: The LLM writes the script as a tool-call argument. Very large scripts may hit tool-call argument size limits or degrade LLM quality.
   - Owner: RFC author
   - Status: Open — estimated practical limit is ~4KB of script (≈1000 tokens). Need to verify against actual model tool-call limits.

6. **Should journals persist to the `SnapshotStore` lifecycle dimension?**
   - Context: RFC-0042 defines `SnapshotStore` as a lifecycle dimension for run state persistence. The journal could use `SnapshotStore` for cross-session resumability.
   - Owner: RFC author
   - Status: Open — deferred to v2. v1 journals are in-memory only.

---

## Decision Record

> Complete this section after RFC review is concluded.

### Decision

**Status**: [APPROVED / REJECTED / DEFERRED]

**Date**: YYYY-MM-DD

**Approvers**:
- [Name 1]
- [Name 2]

### Decision Summary

[Brief statement of the decision made]

### Key Discussion Points

1. [Point 1]
2. [Point 2]

### Conditions of Approval

[Any conditions or modifications required]

### Dissenting Opinions

[Document any significant disagreements for the record]

---

## References

### Related Documents

- [RFC-0055: Dynamic Team Mode](../draft/RFC-0055-dynamic-team-mode.md) — LLM-driven team creation via tools
- [RFC-0042: Unified Lifecycle Architecture](../draft/RFC-0042-unified-lifecycle-architecture.md) — RunLoop, CommChannel, Journal, SnapshotStore
- [RFC-0034: Background Task Redesign](../draft/RFC-0034-background-task-redesign.md) — async task execution
- [openspec/specs/static-graph-workflows/spec.md](../../../openspec/specs/static-graph-workflows/spec.md) — static graph workflow spec
- [src/wolfharness/capabilities/AGENTS.md](../../../src/wolfharness/capabilities/AGENTS.md) — capability conventions

### External Resources

- [Claude Code Dynamic Workflows Documentation](https://code.claude.com/docs/en/workflows)
- [Claude Code Blog: Introducing Dynamic Workflows](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code)
- [Claude Code Blog: A Harness for Every Task](https://claude.com/blog/a-harness-for-every-task-dynamic-workflows-in-claude-code)
- [pi-dynamic-workflows GitHub Repository](https://github.com/michaelliv/pi-dynamic-workflows)
- [Dynamic Workflows in Claude Code: How the Harness Actually Works](https://claudefa.st/blog/guide/development/dynamic-workflows)
- [Claude Code Dynamic Workflows Inside Out](https://www.akshayparkhi.net/2026/May/29/claude-code-dynamic-workflows-inside-out/)
- [Authoring Dynamic Workflows (Claude Lab)](https://claudelab.net/en/articles/claude-code/claude-code-dynamic-workflow-authoring)

### Appendix

#### A. Claude Code Workflow Primitives (for reference)

| Primitive | Description |
|-----------|-------------|
| `agent(prompt, opts)` | Spawn an isolated subagent. Returns text or validated object (with `opts.schema`). |
| `parallel(thunks)` | Run array of `() => agent(...)` concurrently. Barrier — waits for all. Results in input order. |
| `pipeline(items, ...stages)` | Each item flows independently through stages. No cross-item barrier. |
| `phase(title)` | Mark current phase for progress grouping. |
| `log(message)` | Append a workflow-level log line. |
| `args` | Optional JSON value from tool's `args` parameter. |
| `cwd` | Current working directory for subagents. |
| `budget` | `{ total, spent(), remaining() }` token budget tracker. |

#### B. Six Workflow Patterns (from Claude Code)

1. **Fan-out-and-synthesize**: Split task into steps, one agent per step, synthesize results.
2. **Adversarial verification**: For each agent's output, a separate agent adversarially verifies.
3. **Multi-attempt voting**: N agents attempt the same task; judges pick the winner.
4. **Loop until done**: Loop spawning agents until a stop condition is met.
5. **Pipeline**: Items flow through stages independently (no barrier between stages).
6. **Tournament sort**: Pairwise comparisons via agents to rank items.

#### C. pi-dynamic-workflows Determinism Rules

The following are blocked in the pi-dynamic-workflows sandbox (JS `vm`):
- `Date.now()`, `new Date()`
- `Math.random()`
- `require`, `import`, `fs`, network APIs
- Spreads, computed keys, template interpolation, function calls inside `meta`

The Python equivalent (this RFC) blocks:
- `import` statements
- `random`, `time`, `datetime` modules (via name blocking)
- `open`, `eval`, `exec`, `compile`
- All dunder attribute access
- `os`, `sys`, `subprocess`, `socket`, `http`, `urllib` (via name blocking)
