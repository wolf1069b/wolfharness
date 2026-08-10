---
rfc_id: RFC-0034
title: BackgroundTask Architecture Redesign for AgentPool
status: REVIEW
author: yuchen.liu
reviewers:
  - name: TBD
    status: pending
created: 2026-06-09
last_updated: 2026-07-21
decision_date:
related_documents:
  - packages/xeno-agent/docs/rfcs/RFC-0001-async-task-background-task-v2.md
  - .omo/notepads/background-task-provider/learnings.md
  - .omo/notepads/background-task-provider/decisions.md
related_rfcs:
  - RFC-0021 (Agent Concurrent Execution Safety)
  - RFC-0001 (Workers and Teams Session Management)
  - RFC-0026 (Per-Session Agent Isolation)
---

# RFC-0034: BackgroundTask Architecture Redesign for AgentPool

## Table of Contents

- [1. Overview](#1-overview)
- 2. Background & Context
- [3. Problem Statement](#3-problem-statement)
- 4. Goals & Non-Goals
- [5. Evaluation Criteria](#5-evaluation-criteria)
- [6. Design Decisions](#6-design-decisions)
- [7. Technical Design](#7-technical-design)
- [8. API Design](#8-api-design)
- [9. SessionPool Integration](#9-sessionpool-integration)
- [10. Pydantic-AI Design Gap Analysis](#10-pydantic-ai-design-gap-analysis)
- [11. MCP SEP-1686 Compatibility Mapping](#11-mcp-sep-1686-compatibility-mapping)
- [12. Migration Plan](#12-migration-plan)
- [13. Implementation Plan](#13-implementation-plan)
- [14. Open Questions](#14-open-questions)
- [15. Decision Record](#15-decision-record)
- [16. References](#16-references)

---

## 1. Overview

### 1.1 Summary

This RFC proposes a fundamental redesign of the BackgroundTask system, migrating it from the xeno-agent business layer into wolfharness as a first-class core infrastructure, while preserving xeno-agent-specific provider customizations. The redesign replaces the legacy string-based task ID API with a structured `TaskHandle` object pattern aligned with MCP SEP-1686, introduces configurable session-end lifecycle policies, and hard-cuts over from the old API without a compatibility layer.

### 1.2 Why This Matters Now

The current BackgroundTask implementation lives entirely in `xeno-agent` and suffers from several architectural deficiencies discovered during production usage:

1. **Fire-and-forget bug**: The `_notify_parent` callback uses `asyncio.create_task()` without `cancel_and_drain`, causing exceptions to be silently discarded when the parent session has ended.
2. **`parent_session_id` empty string bug**: Prevents auto-resume of background task completion notifications.
3. **String ID API**: The legacy API returns raw strings (`task_id`) with no structured handle, making it impossible to chain operations or attach metadata.
4. **No upstream standard**: Each downstream project reinvents background task patterns; wolfharness lacks a unified abstraction.

These issues block reliable multi-agent delegation workflows and create support burden.

### 1.3 Expected Outcome

After implementation:
- **Unified infrastructure**: `BackgroundTaskManager` and `TaskHandle` are core wolfharness primitives.
- **Structured API**: All background task operations return `TaskHandle` objects with typed methods (`status`, `result()`, `cancel()`).
- **Session lifecycle integration**: Configurable `session_end_policy` controls what happens to background tasks when their parent session ends.
- **SEP-1686 alignment**: API surface is compatible with MCP task semantics, enabling future protocol-native task support.
- **Clean migration**: 124 xeno-agent tests are updated to the new API; no legacy compatibility layer remains.

---

## 2. Background & Context

### 2.1 Current State

The BackgroundTask system currently consists of three layers:

| Layer | Location | Responsibility |
|-------|----------|---------------|
| **Core types** | `xeno_agent/task/types.py` | `BackgroundTask` dataclass, `TaskHandle` dataclass, `TaskStatus` literal |
| **Lifecycle manager** | `xeno_agent/task/manager.py` | `BackgroundTaskManager` — semaphore, timeout, cancellation, cleanup |
| **Provider integration** | `xeno_agent/wolfharness/resource_providers/background_task_provider.py` | Tool definitions (`task`, `background_output`, `background_cancel`) |

The existing `BackgroundTaskProvider` exposes three tools:

```python
async def task(
    self, ctx, mode, message, expected_output="", load_skills=None, title=None, async_mode=False
) -> str

async def background_output(
    self, ctx, task_id, block=False, timeout_seconds=60.0
) -> str

async def background_cancel(
    self, ctx, task_id=None, cancel_all=False
) -> str
```

Key problems with this design:
- `async_mode=False` conflates synchronous delegation with background execution in a single tool.
- Return values are raw strings (or JSON strings), not typed objects.
- `background_output` mixes status querying with blocking result retrieval.
- No connection to `SessionPool` lifecycle — background tasks run orphaned when sessions end.

### 2.2 Historical Context

The BackgroundTask system was initially designed as a xeno-agent-specific augmentation to wolfharness's `subagent_tools.py`. The original RFC (RFC-0001-v2) explicitly stated: "xeno-agent layer优先：先在业务层验证，成熟后考虑 upstream 到 wolfharness." After 3 months of production validation with 130+ tests, the pattern has proven stable enough to upstream.

### 2.3 Glossary

| Term | Definition |
|------|------------|
| **BackgroundTask** | Serializable dataclass representing a task's metadata and lifecycle state |
| **TaskHandle** | Runtime object wrapping an `asyncio.Task` with structured query/cancel methods |
| **TaskStatus** | Literal union: `pending`, `running`, `cancelling`, `completed`, `error`, `cancelled`, `timed_out` |
| **SessionPool** | wolfharness's session orchestration layer (`SessionPool` + `SessionController`) |
| **session_end_policy** | Configuration controlling background task behavior on parent session end |
| **SEP-1686** | MCP Specification Enhancement Proposal for asynchronous task tools |
| **fire-and-forget** | Pattern of spawning async work without awaiting or cleanup; explicitly avoided |
| **cancel_and_drain** | Pydantic-AI utility: cancel tasks and await their cleanup; no orphan tasks |

### 2.4 Related Work

- **RFC-0021**: Introduced `AgentRunContext` for per-call state isolation; BackgroundTask redesign must respect per-run context boundaries.
- **RFC-0026**: Per-session agent isolation ensures background tasks do not share `MessageHistory` across concurrent subagents.
- **RFC-0001 (wolfharness)**: Workers and Teams session management establishes `SpawnSessionStart` patterns that background tasks must emit consistently.

### 2.5 Cross-System Comparison: opencode

A comparative analysis of opencode's background task system (located at `/Users/mollion-mo/Downloads/projects/opencode`) was conducted to inform this RFC's status query design. Key findings:

#### 2.5.1 Tool Surface

opencode exposes a **single** `task` tool with an optional `background: boolean` parameter. There is no `background_output`, `background_cancel`, `task_status`, or `steer_task` tool. A test in `packages/opencode/test/tool/registry.test.ts:103-110` explicitly asserts that `task_status` does **not** exist as a registered tool, confirming this is a deliberate design decision, not an omission.

#### 2.5.2 Status Visibility from the LLM's Perspective

The parent agent's LLM sees only two state transitions:

1. **Launch**: `task(background=true)` returns immediately with `<task id="..." state="running">`.
2. **Completion**: A background fiber calls `inject()` which inserts a synthetic message (`synthetic: true`) into the parent session's prompt stream containing `<task id="..." state="completed|error">` and the task output.

Intermediate status is **invisible** to the LLM. The tool's instruction text explicitly commands: "DO NOT sleep, poll for progress, ask the task for status, or duplicate this task's work."

#### 2.5.3 Internal Status (System-Level Only)

opencode's `BackgroundJob.Service` provides `get(id)` and `list()` methods returning `Info` objects with `status`, `started_at`, `completed_at`, `output`, and `error` fields. `ToolPart` states (`pending`, `running`, `completed`, `error`) propagate via SDK `message.part.updated` events for UI rendering. However, these are **system-level APIs only** — not exposed as LLM-facing tools.

#### 2.5.4 Cancellation

No standalone cancel tool. Cancellation is implicit: when a parent session is cancelled, `SessionRunState.cancelBackgroundJobs` lists running jobs via `background.list()`, matches by `parentSessionId`/`sessionId` metadata, and cancels them recursively.

#### 2.5.5 Comparison Matrix

| Capability | opencode | xeno-agent (current) | This RFC (proposed) |
|-----------|----------|---------------------|---------------------|
| LLM-visible status query tool | None (test-enforced absence) | `background_output(block=False)` returns status summary | `task_status()` tool with structured return |
| Intermediate progress visibility | Not possible for LLM | `output_file` written but not exposed to LLM | `progress_preview` field in `task_status()` return |
| Completion notification | Auto-inject synthetic message | `NotificationBatcher` via `followup()` | `session_end_policy: notify` + `NotificationBatcher` |
| Standalone cancel tool | No (session lifecycle cascade) | `background_cancel` tool | `cancel_task()` tool |
| Steering (mid-task injection) | No | `steer_task` tool | `steer_task()` tool (preserved) |
| Polling guidance | Instruction-level prohibition | Allowed but discouraged | Multi-layer norm constraint (see §7.5) |
| States | 4 (`running`, `completed`, `error`, `cancelled`) | 7 (`pending`, `running`, `cancelling`, `completed`, `error`, `cancelled`, `timed_out`) | 7 (preserved) |

#### 2.5.6 Key Takeaway

opencode's "no status query" design works for its use case (short research tasks, ~30s–5min, cheap to cancel and restart). xeno-agent's industrial diagnosis use case involves longer tasks (5–10min deep analysis) where blind cancellation is expensive. The `steer_task` tool — which xeno-agent has and opencode lacks — requires status information to make informed intervention decisions. Therefore, this RFC proposes **providing** status query capability while **constraining** its usage through multi-layer norm design (see §7.5), rather than following opencode's approach of technical absence.

---

## 3. Problem Statement

### 3.1 Specific Problems

1. **UI Events != LLM Context**: `SubAgentEvent` emitted by background tasks reaches the UI stream via `ctx.events.emit_event()`, but the lead agent's LLM conversation context (its `message_history`) is never updated. The LLM is unaware of task completion unless the user explicitly calls `background_output`.

2. **Fire-and-forget notification bug**: The `_on_task_completed` callback in `BackgroundTaskProvider._task_async()` spawns `_notify_parent()` via `asyncio.create_task()` without `cancel_and_drain`. When `session_pool.inject_prompt()` raises (e.g., parent session ended), the exception is silently lost.

3. **Empty `parent_session_id`**: Line 379 of `background_task_provider.py` falls back to `getattr(ctx.node, 'session_id', '')` — an empty string prevents `SessionPool.inject_prompt()` from routing the notification correctly.

4. **No upstream abstraction**: Every downstream project that wants background tasks must copy xeno-agent's implementation. There is no wolfharness-native primitive.

5. **String-based API**: Returning raw string IDs makes it impossible for the LLM to introspect task state without making another tool call.

6. **No progress visibility for running tasks**: The current `background_output(block=False)` returns only `status` + `started_at`. The `_run_and_stream` coroutine incrementally writes to `output_file` via `fs.pipe()` during execution, but this content is **not read** in non-block mode. The parent agent cannot see what the subagent is currently doing, how much output it has produced, or whether it is making progress. For industrial diagnosis tasks running 5–10 minutes, this opacity makes it impossible to distinguish a stuck task from a productive one, leading to either premature cancellation (wasting minutes of work) or indefinite waiting.

7. **No task listing capability**: There is no tool to list all active background tasks for the current session. The parent agent must track task IDs in its own context, which is fragile — if the LLM loses track of a task ID (common in long conversations), the task result becomes irretrievable.

### 3.2 Evidence

| Metric | Observation | Source |
|--------|-------------|--------|
| diagnosis-planning cancellation rate | 100% tasks marked "cancelled" before `_safe_emit_event` fix | `learnings.md` Section 5 |
| case-document race condition | Intermittent event emission failures | `learnings.md` Section 5 |
| Test coverage | 130 tests pass, but 0 tests verify LLM context injection | `test_background_task_*.py` |
| API ergonomics | LLM must call `background_output(task_id=...)` to check status | User feedback |

### 3.3 Impact of Not Solving

- **Reliability**: Background tasks appear to "disappear" when parent sessions end; users cannot retrieve results.
- **LLM Coordination**: The lead agent has no automatic awareness of background work completion, limiting multi-agent planning.
- **Ecosystem Fragmentation**: Each downstream project builds incompatible background task systems.
- **MCP Incompatibility**: Cannot expose background tasks via MCP because SEP-1686 requires structured handles.

---

## 4. Goals & Non-Goals

### 4.1 Goals (In Scope)

1. **Upstream core infrastructure**: `BackgroundTaskManager`, `TaskHandle`, and `TaskStatus` become wolfharness core primitives.
2. **Structured API**: Replace string-ID returns with `TaskHandle` objects exposing `.status`, `.result()`, `.cancel()`.
3. **Session lifecycle integration**: Introduce `session_end_policy` with `cancel`, `keep`, and `notify` strategies.
4. **SEP-1686 alignment**: API signatures and semantics match MCP task tool patterns.
5. **Hard switch**: Directly replace the old API in xeno-agent; no compatibility shim.
6. **Pydantic-AI compliance**: Every `create_task` must have matching `cancel_and_drain` or `await`.

7. **Status query with progress visibility**: Provide `task_status()` tool that returns structured status including a `progress_preview` field (tail of `output_file`) for running tasks, enabling stuck-task detection without full result retrieval.

8. **Multi-layer norm constraint on status polling**: Status query capability is technically available but constrained through tool description, parameter design, runtime soft-limits, and system prompt directives to prevent misuse (see §7.5).

### 4.2 Non-Goals (Out of Scope)

1. **Not**: Implementing MCP server-side task support (SEP-1686 server changes are a separate effort).
2. **Not**: Persistent task storage across process restarts.
3. **Not**: Distributed task execution across multiple wolfharness instances.
4. **Not**: Changing the underlying LLM provider concurrency model.
5. **Not**: Adding new event types beyond what already exists (`SpawnSessionStart`, `SubAgentEvent`, `StreamCompleteEvent`).

### 4.3 Success Criteria

- [ ] `TaskHandle` is importable from `wolfharness.tasks`.
- [ ] `BackgroundTaskManager` is importable from `wolfharness.tasks`.
- [ ] `session_end_policy` is configurable per `AgentPool` or per session.
- [ ] xeno-agent's `BackgroundTaskProvider` delegates to wolfharness core.
- [ ] All 124 xeno-agent background task tests pass with new API.
- [ ] No `asyncio.create_task()` without matching `cancel_and_drain` pattern.
- [ ] Pydantic-AI `cancel_and_drain` is used in all cleanup paths.

---

## 5. Evaluation Criteria

| Criterion | Weight | Description | Measurement |
|-----------|--------|-------------|-------------|
| **API Ergonomics** | Critical | TaskHandle is intuitive and chainable | Code review + user testing |
| **Backward Compatibility Risk** | High | Hard switch must not break non-test code | All integration tests pass |
| **Implementation Complexity** | Medium | Reasonable effort for upstream + migration | Estimated dev days |
| **Session Lifecycle Correctness** | Critical | Tasks behave correctly on session end | Unit tests for all 3 policies |
| **MCP Alignment** | Medium | SEP-1686 compatibility verified | Signature mapping review |
| **Maintainability** | Medium | Code remains understandable | Code review approval |

---

## 6. Design Decisions

This section documents the four confirmed design decisions that govern the architecture.

### Q1: Where does the BackgroundTask infrastructure live?

**Decision: C — Hybrid approach: wolfharness core infrastructure + xeno-agent specific Provider**

**Rationale**:

The core lifecycle manager (`BackgroundTaskManager`), types (`TaskHandle`, `BackgroundTask`), and session integration belong in wolfharness so all downstream projects benefit. However, the specific tool schemas, prompt formatting, and xeno-agent-specific behaviors (e.g., `load_skills` injection, XML prompt formatting) remain in xeno-agent's `BackgroundTaskProvider`.

**Scope split**:

| Component | wolfharness (core) | xeno-agent (provider) |
|-----------|-----------------|----------------------|
| `TaskHandle` | ✅ | ❌ |
| `BackgroundTask` dataclass | ✅ | ❌ |
| `TaskStatus` literal | ✅ | ❌ |
| `BackgroundTaskManager` | ✅ | ❌ |
| `session_end_policy` integration | ✅ | ❌ |
| `run_background_task()` tool | ❌ | ✅ |
| `task_status()` tool | ❌ | ✅ |
| `task_result()` tool | ❌ | ✅ |
| `cancel_task()` tool | ❌ | ✅ |
| XML prompt formatting | ❌ | ✅ |
| `load_skills` resolution | ❌ | ✅ |

**Trade-offs**:
- **Pro**: Downstream projects (xeno-rag, xeno-serve) can reuse core infrastructure.
- **Pro**: wolfharness can integrate background tasks with `SessionPool` natively.
- **Con**: Slightly more complex import graph; xeno-agent depends on wolfharness tasks.
- **Con**: Core changes require coordinated releases.

### Q2: What is the API pattern for task handles?

**Decision: B — New API pattern: `TaskHandle` replaces string task IDs**

**Old API** (to be replaced):

```python
# Returns a formatted string with task_id buried inside
result = await task(agent_mode="expert", prompt="analyze", async_mode=True)
# result == "Background task launched.\n\nTask ID: bg_abc123\n..."

# Must extract task_id and pass it as a string
status = await background_output(task_id="bg_abc123", block=False)
```

**New API** (SEP-1686 style):

```python
from wolfharness.tasks import TaskHandle

# Returns a structured handle
handle: TaskHandle = await run_background_task(agent_mode="expert", prompt="analyze")

# Introspect without additional tool calls
print(handle.status)        # "running"
print(handle.task_id)       # "bg_abc123"

# Retrieve result when complete
result = await handle.result(timeout=60.0)

# Cancel explicitly
await handle.cancel()
```

**Trade-offs**:
- **Pro**: Type-safe; IDE autocomplete works.
- **Pro**: Natural chaining; no string parsing.
- **Pro**: Aligns with MCP SEP-1686 `TaskHandle` semantics.
- **Con**: Requires updating all call sites (124 tests).
- **Con**: LLM tool schemas must describe object fields instead of string returns.

### Q3: What happens to background tasks when a session ends?

**Decision: C — Configurable policy: `session_end_policy = cancel | keep | notify`**

**Policies**:

| Policy | Behavior | Use Case |
|--------|----------|----------|
| `cancel` | Cancel all associated background tasks on session end | Default; prevents resource leaks |
| `keep` | Tasks continue running after session end | Long-running analysis jobs |
| `notify` | Inject completion prompt into next session turn, then continue | Interactive workflows where user must see results |

**Rationale**:

A single hardcoded behavior cannot satisfy all use cases. Industrial diagnostics (`xeno-agent`) often needs `notify` so the fault expert receives completion prompts. Batch processing jobs need `keep`. Default wolfharness behavior should be `cancel` for safety.

**Implementation hook**:

```python
# In SessionController.close_session()
policy = session.config.get("session_end_policy", "cancel")
for task in task_manager.get_tasks_by_session(session_id):
    match policy:
        case "cancel":
            await task.cancel()
        case "keep":
            continue
        case "notify":
            task.set_on_completed(lambda t: self._inject_prompt_next_turn(session_id, t))
```

**Trade-offs**:
- **Pro**: Flexible; covers all known use cases.
- **Pro**: Backward-compatible default (`cancel`) is safest.
- **Con**: `notify` policy has the "lead agent turn ended" problem (see Section 10).

### Q4: Should we maintain a compatibility layer?

**Decision: A — Hard cutover: directly replace the API, no compatibility layer**

**Rationale**:

- The old API has 3 methods (`task`, `background_output`, `background_cancel`) with 1,215 lines in a single provider file.
- A compatibility layer would require maintaining both string-ID and TaskHandle paths, doubling testing surface.
- xeno-agent is the only known consumer; we control all 124 tests.
- A clean break reduces long-term maintenance burden.

**Migration scope**:
- `xeno-agent` tests: 124 test cases updated.
- `xeno-agent` provider: `BackgroundTaskProvider` rewritten to delegate to wolfharness core.
- `diag-agent.yaml`: Tool schema references updated.

**Trade-offs**:
- **Pro**: Cleanest long-term API surface.
- **Pro**: No dual-path maintenance.
- **Con**: All consumers must migrate atomically.
- **Con**: Cannot partially upgrade.

### Q5: How should running-task status and progress be exposed to the LLM?

**Decision: B — Provide structured status query with progress preview, constrained by multi-layer norms**

Three options were considered:

| Option | Description | LLM Can Query Status? | LLM Can See Progress? | Polling Risk |
|--------|-------------|----------------------|----------------------|--------------|
| A: opencode-style (no status tool) | No `task_status` tool; rely solely on auto-notification | No | No | None (impossible) |
| B: Status + progress with norms | `task_status()` returns status + `progress_preview`; usage constrained by §7.5 | Yes | Yes (tail of output) | Mitigated by multi-layer constraints |
| C: Status only, no progress | `task_status()` returns status + timing only | Yes | No | Medium (status-only is less useful, may tempt repeated polling) |

**Rationale for B**:

1. **Steering requires status**: xeno-agent has a `steer_task` tool that opencode lacks. Making informed steering decisions (intervene vs. cancel vs. wait) requires knowing the task's current state and progress. Option A makes `steer_task` blind.

2. **Industrial diagnosis cost asymmetry**: A 5–10 minute diagnostic task that is blindly cancelled loses significant work. The ability to distinguish "stuck" from "productive" via `progress_preview` justifies the polling risk.

3. **Data already exists**: `_run_and_stream` already incrementally writes to `output_file` via `fs.pipe()`. The `progress_preview` field reads the tail of this file — no new data pipeline needed, just exposing existing data.

4. **Norms mitigate risk**: The multi-layer constraint design (§7.5) — tool description, `status_only` parameter, runtime soft-limits, system prompt — channels usage toward the intended pattern (check status only when stuck or before steering) without making the capability technically impossible.

**Trade-offs**:
- **Pro**: Enables informed steering and stuck-task detection.
- **Pro**: `progress_preview` provides actionable signal, not just "still running."
- **Con**: LLM may poll despite norms, consuming context window.
- **Con**: More complex than opencode's clean "no status" approach.

---

## 7. Technical Design

### 7.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AgentPool Core                                  │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────┐  │
│  │  SessionController  │───▶│ BackgroundTaskManager│───▶│   TaskHandle    │  │
│  └─────────────────────┘    └─────────────────────┘    └─────────────────┘  │
│           │                          │                                      │
│           │ session_end_policy       │ registry                               │
│           ▼                          ▼                                      │
│  ┌─────────────────────┐    ┌─────────────────────┐                        │
│  │   SessionConfig     │    │   BackgroundTask    │                        │
│  │  (cancel|keep|notify)│    │   (serializable)    │                        │
│  └─────────────────────┘    └─────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ▲
                                       │ delegates core
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Xeno-Agent Provider                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    BackgroundTaskProvider                            │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │    │
│  │  │ run_background_ │  │   task_status   │  │    cancel_task      │  │    │
│  │  │    task()       │  │                 │  │                     │  │    │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │    │
│  │  ┌─────────────────┐  ┌─────────────────┐                           │    │
│  │  │   task_result() │  │  (future: list_ │                           │    │
│  │  │                 │  │    tasks())     │                           │    │
│  │  └─────────────────┘  └─────────────────┘                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Component Design

#### 7.2.1 `wolfharness.tasks` Module (New)

```python
# wolfharness/tasks/__init__.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
import asyncio

TaskStatus = Literal[
    "pending", "running", "cancelling",
    "completed", "error", "cancelled", "timed_out",
]

TERMINAL_STATES: frozenset[TaskStatus] = frozenset(
    {"completed", "error", "cancelled", "timed_out"}
)


@dataclass(slots=True)
class BackgroundTask:
    """Serializable representation of a background task."""

    id: str
    description: str
    agent_name: str
    prompt: str
    parent_session_id: str | None
    child_session_id: str | None
    status: TaskStatus = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(datetime.timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: str | None = None
    error: str | None = None


@dataclass(slots=True)
class TaskHandle:
    """Runtime handle for a background task.

    Provides structured access to task state and lifecycle operations.
    Aligned with MCP SEP-1686 TaskHandle semantics.
    """

    task_id: str
    _manager: "BackgroundTaskManager"

    @property
    def status(self) -> TaskStatus:
        """Return the current task status."""
        task = self._manager.get_task(self.task_id)
        if task is None:
            return "error"
        return task.status

    async def result(self, timeout: float | None = None) -> str:
        """Block until the task completes and return its result.

        Args:
            timeout: Maximum seconds to wait. If None, waits indefinitely.

        Returns:
            The task result string.

        Raises:
            TaskError: If the task failed, was cancelled, or timed out.
            TimeoutError: If the wait exceeded ``timeout``.
        """
        ...

    async def cancel(self) -> bool:
        """Request cancellation of this task.

        Returns:
            True if cancellation was initiated, False if task was already terminal.
        """
        ...
```

#### 7.2.2 `BackgroundTaskManager` (wolfharness core)

```python
# wolfharness/tasks/manager.py
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from pydantic_ai._utils import cancel_and_drain

from .types import BackgroundTask, TaskHandle, TaskStatus, TERMINAL_STATES


class BackgroundTaskManager:
    """Manages background task lifecycle, concurrency, timeout, and cleanup.

    This is the wolfharness-core equivalent of xeno-agent's
    ``BackgroundTaskManager``, with these key differences:

    1. Uses ``cancel_and_drain`` from pydantic-ai for all cleanup paths.
    2. Integrates with ``SessionController`` for session-end policy enforcement.
    3. Exposes ``TaskHandle`` objects instead of raw string IDs.
    """

    def __init__(
        self,
        timeout_seconds: float = 1800.0,
        max_concurrent_tasks: int = 5,
        cleanup_after_seconds: float = 3600.0,
        cancel_timeout_seconds: float = 30.0,
    ) -> None:
        self._tasks: dict[str, BackgroundTask] = {}
        self._handles: dict[str, "_InternalHandle"] = {}
        self._concurrency_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._timeout_seconds = timeout_seconds
        self._cleanup_after_seconds = cleanup_after_seconds
        self._cancel_timeout_seconds = cancel_timeout_seconds
        self._session_tasks: dict[str, set[str]] = {}  # session_id -> {task_id}

    def create_task(
        self,
        description: str,
        agent_name: str,
        prompt: str,
        coro: Any,
        parent_session_id: str | None = None,
        child_session_id: str | None = None,
        on_completed: Callable[[BackgroundTask], None] | None = None,
    ) -> TaskHandle:
        """Register and start a background task.

        Returns a ``TaskHandle`` for structured interaction.
        The underlying asyncio.Task is created immediately but may wait
        on the concurrency semaphore before executing ``coro``.
        """
        ...

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task by ID."""
        ...

    async def cancel_all(self, session_id: str | None = None) -> int:
        """Cancel all non-terminal tasks, optionally scoped to a session."""
        ...

    def get_task(self, task_id: str) -> BackgroundTask | None:
        ...

    def get_tasks_by_session(self, session_id: str) -> list[BackgroundTask]:
        ...

    async def shutdown(self) -> None:
        """Cancel all tasks and await cleanup using ``cancel_and_drain``."""
        ...
```

### 7.3 Key Implementation Patterns

#### Cancel-and-Drain Compliance

Every path that creates an `asyncio.Task` must have a matching cleanup path:

```python
# CORRECT: Task created with cancel_and_drain cleanup
task = asyncio.create_task(coro)
try:
    await asyncio.wait_for(task, timeout=timeout)
finally:
    await cancel_and_drain(task)

# CORRECT: Task stored in manager, cleaned up on shutdown
self._running_tasks.add(asyncio.create_task(coro))
# ... later in shutdown() ...
await cancel_and_drain(*self._running_tasks)

# INCORRECT: Fire-and-forget (old xeno-agent bug)
asyncio.create_task(_notify_parent())  # Exception silently lost
```

#### Session Association

Tasks are associated with sessions at creation time:

```python
def create_task(..., parent_session_id: str | None = None, ...) -> TaskHandle:
    task = BackgroundTask(..., parent_session_id=parent_session_id, ...)
    self._tasks[task.id] = task
    if parent_session_id:
        self._session_tasks.setdefault(parent_session_id, set()).add(task.id)
    ...
```

### 7.4 Status Query & Progress Design

#### 7.4.1 The `task_status()` Tool

The `task_status()` tool replaces the non-blocking path of the old `background_output(block=False)`. It returns a structured dictionary (not a raw string) with the following fields:

| Field | Type | Present When | Description |
|-------|------|-------------|-------------|
| `task_id` | `str` | Always | The task identifier |
| `status` | `TaskStatus` | Always | One of the 7 states |
| `description` | `str` | Always | Task description (for identification) |
| `agent_name` | `str` | Always | Which agent is executing |
| `created_at` | `str` (ISO 8601) | Always | Creation timestamp |
| `started_at` | `str` (ISO 8601) \| `null` | Always | Start timestamp, or `null` if pending |
| `duration` | `str` | Running or terminal | Human-readable elapsed time (e.g., "2m 30s") |
| `progress_preview` | `str` \| `null` | Running only | Tail (last ~500 chars) of `output_file` content |
| `error` | `str` \| `null` | Terminal error/timed_out | Error message if applicable |
| `completed_at` | `str` (ISO 8601) \| `null` | Terminal | Completion timestamp |

#### 7.4.2 `progress_preview` Implementation

The `_run_and_stream` coroutine already incrementally writes to `output_file` via `fs.pipe()`. The `progress_preview` field reads the tail of this file when the task is in a non-terminal state:

```python
def _get_progress_preview(self, ctx: AgentContext, task: BackgroundTask) -> str | None:
    """Read the tail of the task's output_file for progress indication."""
    if task.status not in ("running", "pending"):
        return None
    if not task.output_file:
        return None
    try:
        raw = ctx.internal_fs.cat(task.output_file)
        content = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        if not content.strip():
            return None
        # Return last 500 characters to keep context window impact bounded
        return content[-500:] if len(content) > 500 else content
    except Exception:
        return None
```

**Context window budget**: The `progress_preview` is capped at 500 characters. A single `task_status()` call on a running task adds at most ~500 chars + structured fields (~300 chars) = ~800 chars to the LLM context. This is bounded and predictable.

#### 7.4.3 `list_tasks()` — Session Task Enumeration

A companion tool `list_tasks()` (optional, provider-level) returns a summary table of all tasks for the current session:

```python
async def list_tasks(ctx: AgentContext) -> str:
    """List all background tasks for the current session.

    Returns a markdown table with task_id, description, status, and duration.
    Useful for recovering lost task IDs in long conversations.
    """
    ...
```

This addresses Problem 7 (§3.1): the LLM losing track of task IDs in long conversations. The return is a compact markdown table, not full task details, keeping context impact minimal.

### 7.5 Multi-Layer Norm Constraint Design

The status query capability is **technically available but constrained** through four layers. Each layer independently reduces the probability of misuse; together they form a defense-in-depth against polling anti-patterns.

#### Layer 1: Tool Description (Static)

The `task_status()` tool's description in the YAML schema explicitly scopes its intended use:

```yaml
# task_status.yaml
description: |
  Query the current status of a background task, including a progress preview
  for running tasks.

  Use this tool ONLY when:
  - You suspect a task may be stuck (no notification received after expected duration)
  - You need status information before deciding to steer or cancel a task

  Do NOT use this tool for routine progress checking. You will receive an
  automatic notification when each task completes — wait for it.
```

This is the first signal the LLM sees. It defines the social contract: the tool exists for stuck-detection and steering-decisions, not for progress monitoring.

#### Layer 2: Parameter Design (Structural)

The `task_status()` tool separates "query status" from "retrieve result" into distinct tools (`task_status` vs `task_result`). This prevents the pattern where `background_output(block=False)` was used for both purposes, making casual status checks feel like a natural part of result retrieval.

Additionally, the `progress_preview` field is **always included** when the task is running — there is no opt-in parameter. This means the LLM gets a useful snapshot in a single call, reducing the temptation to call repeatedly for "just a bit more" output.

#### Layer 3: Runtime Soft-Limit (Advisory)

The provider tracks `task_status()` call frequency per task per session. When the LLM calls `task_status()` for the same task more than `N` times within `M` seconds (configurable, defaults: N=3, M=60), the return includes an additional warning field:

```python
{
    "task_id": "bg_abc123",
    "status": "running",
    "progress_preview": "...",
    "_warning": "You have queried this task 3 times in the last 60 seconds. "
                "The task is still running. Consider waiting for the completion "
                "notification instead of polling."
}
```

This is advisory, not blocking — the tool still returns the status. The warning is injected into the structured return, ensuring the LLM sees it in the same response that tempted it to poll.

#### Layer 4: System Prompt Directive (Instructional)

The agent's system prompt includes a directive about background task etiquette:

```
## Background Task Etiquette

When you launch background tasks:
1. Continue with other work while tasks run — do not wait idly.
2. You will receive a <system-reminder> notification when each task completes.
3. Use `task_status()` ONLY if you suspect a task is stuck or before steering/cancelling.
4. Do NOT poll task_status() repeatedly for progress updates.
```

This is the same mechanism opencode uses (tool instruction text prohibiting polling), adapted to coexist with the technical capability.

#### Constraint Effectiveness Analysis

| Layer | What It Prevents | Bypass Risk | Cost of Implementation |
|-------|-----------------|-------------|----------------------|
| Tool description | Unaware misuse (LLM doesn't know it shouldn't poll) | High (LLM may ignore) | Low (YAML text) |
| Parameter design | Conflating status check with result retrieval | Medium (separate tools reduce temptation) | Low (API design) |
| Runtime soft-limit | Sustained polling loops | Low (warning is in-band, hard to ignore) | Medium (provider state tracking) |
| System prompt | All forms of polling | Medium (instruction-following varies by model) | Low (prompt text) |

No single layer is sufficient. Together, they create a gradient from "the LLM knows it shouldn't poll" (Layer 1+4) to "the system makes polling less rewarding" (Layer 2) to "the system actively warns when polling is detected" (Layer 3).

---

## 8. API Design

### 8.1 wolfharness Core API

```python
# wolfharness/tasks/__init__.py

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Any

TaskStatus = Literal[
    "pending", "running", "cancelling",
    "completed", "error", "cancelled", "timed_out",
]


@dataclass(frozen=True, slots=True)
class TaskHandle:
    """Structured handle for an asynchronous background task.

    Aligned with MCP SEP-1686 TaskHandle semantics.
    """

    task_id: str

    @property
    def status(self) -> TaskStatus: ...

    async def result(self, timeout: float | None = None) -> str: ...

    async def cancel(self) -> bool: ...


class BackgroundTaskManager:
    def __init__(
        self,
        *,
        timeout_seconds: float = 1800.0,
        max_concurrent_tasks: int = 5,
        cleanup_after_seconds: float = 3600.0,
        cancel_timeout_seconds: float = 30.0,
    ) -> None: ...

    def create_task(
        self,
        *,
        description: str,
        agent_name: str,
        prompt: str,
        coro: Any,
        parent_session_id: str | None = None,
        child_session_id: str | None = None,
        on_completed: Any | None = None,
    ) -> TaskHandle: ...

    async def cancel_task(self, task_id: str) -> bool: ...

    async def cancel_all(self, session_id: str | None = None) -> int: ...

    def get_task(self, task_id: str) -> BackgroundTask | None: ...

    def get_tasks_by_session(self, session_id: str) -> list[BackgroundTask]: ...

    async def shutdown(self) -> None: ...
```

### 8.2 xeno-agent Provider API (New Tools)

```python
# xeno_agent/wolfharness/resource_providers/background_task_provider.py

async def run_background_task(
    ctx: AgentContext,
    agent_mode: str,
    prompt: str,
    expected_output: str = "",
    load_skills: list[str] | None = None,
    title: str | None = None,
) -> TaskHandle:
    """Launch a background task and return a structured handle.

    Replaces the old ``task(async_mode=True)`` pattern.
    """
    ...

async def task_status(
    ctx: AgentContext,
    task_id: str,
) -> dict[str, Any]:
    """Query the status of a background task.

    Returns a JSON-serializable dict with status, duration, progress preview,
    and metadata. For running tasks, includes a ``progress_preview`` field
    containing the tail (~500 chars) of the task's incremental output.

    Intended use cases:
    - Stuck-task detection (no notification after expected duration)
    - Pre-steering status check (before calling ``steer_task``)
    - Pre-cancellation status check (before calling ``cancel_task``)

    NOT intended for routine progress polling — wait for the completion
    notification instead.
    """
    ...

async def task_result(
    ctx: AgentContext,
    task_id: str,
    timeout_seconds: float = 60.0,
) -> str:
    """Retrieve the result of a background task, optionally blocking.

    Replaces ``background_output(block=True)``. For non-terminal tasks,
    blocks up to ``timeout_seconds`` for completion. If the timeout fires,
    returns a status message indicating the task is still running.
    """
    ...

async def cancel_task(
    ctx: AgentContext,
    task_id: str | None = None,
    cancel_all: bool = False,
) -> str:
    """Cancel a background task or all tasks.

    Replaces ``background_cancel()``.
    """
    ...

async def steer_task(
    ctx: AgentContext,
    task_id: str,
    message: str,
    mode: Literal["interrupt", "advisory"] = "advisory",
) -> str:
    """Send a steering message to a running background task.

    Preserved from the current implementation. Two modes:
    - ``advisory`` (default): Queued for the next turn.
    - ``interrupt``: Injected into the active turn immediately.
    """
    ...

async def list_tasks(
    ctx: AgentContext,
) -> str:
    """List all background tasks for the current session.

    Returns a markdown table with task_id, description, status, and duration.
    Useful for recovering lost task IDs in long conversations.
    """
    ...
```

### 8.3 YAML Configuration

```yaml
# config/diag-agent.yaml
agents:
  fault_expert:
    type: native
    model: "openai-chat:svc/glm-4.7"
    session_end_policy: notify  # cancel | keep | notify
    tools:
      - type: custom
        import_path: xeno_agent.wolfharness.resource_providers.background_task_provider.XenoBackgroundTaskProvider
        enabled_tools:
          - run_background_task
          - task_status
          - task_result
          - cancel_task
```

---

## 9. SessionPool Integration

### 9.1 Session-End Policy Enforcement

The `SessionController.close_session()` method is the hook point for policy enforcement:

```python
# wolfharness/sessions/controller.py

from wolfharness.tasks import BackgroundTaskManager

class SessionController:
    def __init__(self, ..., task_manager: BackgroundTaskManager):
        self._task_manager = task_manager

    async def close_session(self, session_id: str) -> None:
        policy = self._get_session_end_policy(session_id)
        task_ids = self._task_manager.get_tasks_by_session(session_id)

        match policy:
            case "cancel":
                for task_id in task_ids:
                    if task_id.status not in TERMINAL_STATES:
                        await self._task_manager.cancel_task(task_id)
            case "keep":
                # Detach from session but continue running
                self._task_manager.detach_from_session(session_id)
            case "notify":
                for task_id in task_ids:
                    if task_id.status not in TERMINAL_STATES:
                        # Set up completion notification for next turn
                        self._task_manager.set_notification_callback(
                            task_id,
                            lambda t: self._inject_completion_prompt(session_id, t),
                        )
```

### 9.2 Policy Behavior Matrix

| Policy | Session End Action | Task Continues? | LLM Notified? | Resource Leak Risk |
|--------|-------------------|-----------------|---------------|-------------------|
| `cancel` | Cancel all tasks | No | N/A (task dead) | Lowest |
| `keep` | Detach, no action | Yes | No | Medium (orphaned tasks) |
| `notify` | Set completion callback | Yes | Yes (on complete) | Low |

### 9.3 Notification Injection Path

For the `notify` policy, the completion callback must use `PromptInjectionManager.queue()`:

```python
async def _inject_completion_prompt(
    self,
    session_id: str,
    task: BackgroundTask,
) -> None:
    """Inject a background task completion notice into the session."""
    session = self._sessions.get(session_id)
    if session is None or session.run_ctx is None:
        return

    notice = self._format_completion_notice(task)
    session.run_ctx.injection_manager.queue(notice)
```

**Critical constraint**: If the lead agent's `run_stream` has already exited, the queued prompt is never consumed. This is architecturally correct — the LLM should not receive unprompted context in a new conversation turn without the user's awareness. The task output remains available via `task_result()`.

---

## 10. Pydantic-AI Design Gap Analysis

### 10.1 Pydantic-AI Principle: "No Fire-and-Forget"

Pydantic-AI strictly follows a cleanup discipline: every `asyncio.create_task()` must either be awaited or cancelled-and-drained. The `cancel_and_drain` utility in `pydantic_ai._utils` embodies this:

```python
async def cancel_and_drain(*tasks: asyncio.Task[Any], msg: object = None) -> None:
    """Cancel any tasks still running and wait for them to finish unwinding.

    Cleanup-only: results and exceptions from `tasks` are intentionally
    discarded so a cancelled child cannot replace an exception already
    propagating in the caller.
    """
    for task in tasks:
        if not task.done():
            task.cancel(msg=msg)

    with anyio.CancelScope(shield=True):
        await asyncio.gather(*tasks, return_exceptions=True)
```

**Key design gap**: The current xeno-agent `_notify_parent()` callback violates this principle:

```python
# OLD (BUGGY):
_notify_task = asyncio.create_task(_notify_parent())
_notify_task.add_done_callback(_on_notify_done)
# If _notify_parent() raises, exception goes to _on_notify_done,
# but if the event loop is shutting down, the callback may never fire.
```

**Fix**: Do not use fire-and-forget for notification. Instead:
1. Store the notification task in the manager's cleanup set.
2. On session end or manager shutdown, `cancel_and_drain` all pending notifications.

### 10.2 `StreamedRunResult._on_complete` Pattern

Pydantic-AI's `StreamedRunResult` uses an async callback `_on_complete` that is **awaited** inside the run loop:

```python
# pydantic_ai result handling
if self._on_complete is not None:
    await self._on_complete(self)  # Awaited, not fire-and-forget
```

The BackgroundTaskManager's `on_completed` callback should follow this pattern — it must be awaited, not spawned as a detached task.

### 10.3 Required Changes for Compliance

| Location | Current Pattern | Required Pattern |
|----------|----------------|------------------|
| `_notify_parent()` in provider | `asyncio.create_task()` | Store in manager; `cancel_and_drain` on cleanup |
| `BackgroundTaskManager.shutdown()` | `await handle.task` with suppress | `cancel_and_drain(*all_tasks)` |
| `_execute_task()` in manager | Bare `asyncio.create_task()` | Track in `_running_tasks` set |

---

## 11. MCP SEP-1686 Compatibility Mapping

### 11.1 SEP-1686 Overview

SEP-1686 introduces asynchronous task tools to MCP:

```python
# MCP client usage (SEP-1686)
@server.tool(task=True)
async def long_running_analysis(query: str) -> TaskHandle:
    ...

# TaskHandle semantics (SEP-1686)
handle = await client.call_tool("long_running_analysis", {"query": "..."})
result = await handle.result()   # Block until complete
status = handle.status            # Query current status
await handle.cancel()             # Request cancellation
```

### 11.2 wolfharness-to-SEP-1686 Mapping

| SEP-1686 Concept | wolfharness Equivalent | Notes |
|-----------------|----------------------|-------|
| `@tool(task=True)` | `run_background_task()` tool | Explicit tool, not decorator |
| `TaskHandle.result()` | `TaskHandle.result()` | Direct equivalent |
| `TaskHandle.status` | `TaskHandle.status` | Direct equivalent |
| `TaskHandle.cancel()` | `TaskHandle.cancel()` | Direct equivalent |
| Graceful degradation | `BackgroundTaskManager` fallback | Falls back to sync if no task support |

### 11.3 Future MCP Server Integration

When wolfharness implements an MCP server with SEP-1686 support:

```python
# Future: wolfharness MCP server
from wolfharness.tasks import BackgroundTaskManager

class AgentPoolMCPServer:
    def __init__(self, task_manager: BackgroundTaskManager):
        self._task_manager = task_manager

    async def handle_tool_call(self, tool_name: str, params: dict) -> Any:
        if tool_name == "run_background_task":
            handle = self._task_manager.create_task(...)
            return {
                "_meta": {"task": True},
                "task_id": handle.task_id,
                "status": handle.status,
            }
```

The `TaskHandle` abstraction ensures that the same core object works for both internal tools and external MCP clients.

---

## 12. Migration Plan

### 12.1 xeno-agent Test Migration (124 Tests)

The following test files in `xeno-agent` must be updated:

| Test File | Tests | Migration Actions |
|-----------|-------|-------------------|
| `test_background_task_provider.py` | ~40 | Replace `task(async_mode=True)` with `run_background_task()`; expect `TaskHandle` |
| `test_background_task_cancellation.py` | ~25 | Replace `background_cancel()` with `handle.cancel()` |
| `test_background_task_output.py` | ~20 | Replace `background_output()` with `task_status()` / `task_result()` |
| `test_background_task_events.py` | ~15 | Update event assertions for new spawn patterns |
| `test_background_task_history_isolation.py` | 6 | No logic change; update fixture/setup |
| `test_background_task_cancellation_regression.py` | 13 | Update to `TaskHandle.cancel()` |
| Integration tests | ~5 | Update YAML configs, tool names |

### 12.2 Migration Checklist

- [ ] **Phase 0**: Create wolfharness `tasks` module with `TaskHandle`, `BackgroundTask`, `BackgroundTaskManager`.
- [ ] **Phase 0**: Port `BackgroundTaskManager` from xeno-agent to wolfharness, adding `cancel_and_drain` compliance.
- [ ] **Phase 1**: Add `session_end_policy` to `SessionConfig` and enforcement in `SessionController`.
- [ ] **Phase 2**: Rewrite xeno-agent `BackgroundTaskProvider`:
  - [ ] Replace `task()` with `run_background_task()` returning `TaskHandle`.
  - [ ] Replace `background_output()` with `task_status()` + `task_result()`.
  - [ ] Replace `background_cancel()` with `cancel_task()`.
  - [ ] Fix `_notify_parent()` to use `PromptInjectionManager.queue()` without fire-and-forget.
  - [ ] Fix `parent_session_id` fallback to use `None` instead of empty string.
- [ ] **Phase 3**: Update all 124 xeno-agent tests:
  - [ ] Replace string `task_id` extractions with `handle.task_id`.
  - [ ] Replace `background_output(task_id=...)` with `task_result(task_id=...)`.
  - [ ] Replace `background_cancel(task_id=...)` with `cancel_task(task_id=...)`.
  - [ ] Add `TaskHandle.status` assertions.
  - [ ] Update schema YAMLs to reference new tool names.
- [ ] **Phase 4**: Update `diag-agent.yaml` tool registrations.
- [ ] **Phase 5**: Run full xeno-agent test suite; fix regressions.
- [ ] **Phase 6**: Delete old `xeno_agent/task/manager.py` and `xeno_agent/task/types.py` (or deprecate).

### 12.3 API Mapping (Old -> New)

| Old API | New API | Return Type Change |
|---------|---------|-------------------|
| `task(mode, prompt, async_mode=True)` | `run_background_task(agent_mode, prompt)` | `str` -> `TaskHandle` |
| `background_output(task_id, block=False)` | `task_status(task_id)` (non-blocking) | `str` -> `dict` |
| `background_output(task_id, block=True)` | `task_result(task_id, timeout_seconds=...)` | `str` -> `str` |
| `background_cancel(task_id)` | `cancel_task(task_id)` | `str` -> `str` |
| `background_cancel(cancel_all=True)` | `cancel_task(cancel_all=True)` | `str` -> `str` |

---

## 13. Implementation Plan

### Phase 0: wolfharness Core (Week 1)

**Deliverables**:
- `wolfharness/tasks/types.py` — `TaskStatus`, `BackgroundTask`, `TaskHandle`
- `wolfharness/tasks/manager.py` — `BackgroundTaskManager` with `cancel_and_drain`
- `wolfharness/tasks/__init__.py` — Public exports
- Unit tests for core manager (semaphore, timeout, cancellation, cleanup)

**Files Modified**:
- `src/wolfharness/tasks/` (new directory)
- `src/wolfharness/sessions/controller.py` (session_end_policy hook)
- `src/wolfharness/sessions/config.py` (session_end_policy field)

**Rollback**: Delete `src/wolfharness/tasks/` directory.

### Phase 1: Session Integration (Week 1-2)

**Deliverables**:
- `SessionConfig.session_end_policy: Literal["cancel", "keep", "notify"]`
- `SessionController.close_session()` policy enforcement
- `BackgroundTaskManager` session association methods

**Files Modified**:
- `src/wolfharness/sessions/config.py`
- `src/wolfharness/sessions/controller.py`
- `src/wolfharness/tasks/manager.py`

**Rollback**: Revert session controller changes.

### Phase 2: xeno-agent Provider Rewrite (Week 2-3)

**Deliverables**:
- Rewritten `BackgroundTaskProvider` with 4 new tools
- `_notify_parent()` using `PromptInjectionManager.queue()`
- Empty string `parent_session_id` fix

**Files Modified**:
- `src/xeno_agent/wolfharness/resource_providers/background_task_provider.py`

**Rollback**: Restore from git.

### Phase 3: Test Migration (Week 3-4)

**Deliverables**:
- All 124 tests updated to new API
- New tests for `session_end_policy` behaviors
- New tests for `TaskHandle` methods

**Files Modified**:
- `tests/wolfharness/resource_providers/test_background_task_*.py`

**Rollback**: Restore from git.

### Dependencies

- wolfharness Phase 0 must complete before xeno-agent Phase 2.
- wolfharness Phase 1 must complete before xeno-agent test additions for `session_end_policy`.

---

## 14. Open Questions

1. **Should `TaskHandle` be hashable and comparable?**
   - Context: SEP-1686 does not specify equality semantics.
   - Owner: API design
   - Status: Open

2. **How should `keep` policy tasks be garbage-collected?**
   - Context: Tasks detached from sessions may run indefinitely.
   - Owner: Infrastructure
   - Status: Open

3. **Should `notify` policy support batching (multiple tasks completing together)?**
   - Context: Parallel tasks often complete near-simultaneously.
   - Owner: xeno-agent
   - Status: Open

4. **What is the performance impact of `cancel_and_drain` vs. bare `task.cancel()`?**
   - Context: `cancel_and_drain` shields with `anyio.CancelScope`, adding overhead.
   - Owner: Performance
   - Status: Open

5. **Should wolfharness expose a `@background_task` decorator for tool functions?**
   - Context: Would allow declarative background task registration.
   - Owner: API design
   - Status: Open

6. **What is the optimal `progress_preview` character limit?**
   - Context: 500 chars is a preliminary estimate balancing context window impact vs. usefulness. Too small and it's useless; too large and it pollutes context.
   - Owner: API design
   - Status: Open — needs empirical testing with real diagnostic tasks.

7. **Should the runtime soft-limit (Layer 3 of §7.5) be configurable per-agent?**
   - Context: Different agents may have different polling tolerance. A fast research agent might tolerate N=2; a slow diagnostic agent might allow N=5.
   - Owner: xeno-agent
   - Status: Open

8. **Should `list_tasks()` be a core wolfharness tool or a xeno-agent provider tool?**
   - Context: Task listing is useful for any downstream project, but the markdown table format is presentation-level.
   - Owner: API design
   - Status: Open

9. **How should `progress_preview` handle binary or non-UTF-8 output?**
   - Context: `fs.pipe()` writes bytes; tasks producing binary output (e.g., image analysis) may not have meaningful text previews.
   - Owner: Implementation
   - Status: Open — current design returns `None` on decode failure, which may need refinement.

---

## 15. Decision Record

**Status**: DRAFT (awaiting review)

**Date**: 2026-06-09

**Approvers**: TBD

### Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| D1 | Hybrid core + provider split (Q1=C) | Reuse core, preserve xeno-agent specifics |
| D2 | TaskHandle replaces string IDs (Q2=B) | Type safety, SEP-1686 alignment, ergonomics |
| D3 | Configurable session_end_policy (Q3=C) | Covers all known use cases; safe default |
| D4 | Hard cutover, no compat layer (Q4=A) | Cleanest long-term API; controlled test scope |
| D5 | Status + progress with multi-layer norms (Q5=B) | Enables steering decisions and stuck-task detection; norms mitigate polling risk |

### Key Discussion Points

1. **Why not Option A (all in wolfharness)?** Would force xeno-agent's XML formatting and skill-loading into core, creating inappropriate coupling.
2. **Why not Option B (all in xeno-agent)?** Defeats the purpose of upstreaming proven infrastructure.
3. **Why not keep string IDs?** String parsing is error-prone and incompatible with MCP SEP-1686.
4. **Why not a compatibility layer?** Doubles maintenance burden for a single-consumer API.

### Conditions on Approval

- [ ] At least 2 code reviewers approve
- [ ] wolfharness core tests demonstrate 100% manager coverage
- [ ] All 124 xeno-agent tests pass with new API
- [ ] Session-end policy tests cover all 3 variants
- [ ] No `asyncio.create_task()` without matching cleanup in changed code
- [ ] Documentation updated (wolfharness tasks module, xeno-agent migration guide)

---

## 16. References

### Related Documents

1. **RFC-0001-v2 (xeno-agent)** — Original background task RFC
   Location: `packages/xeno-agent/docs/rfcs/RFC-0001-async-task-background-task-v2.md`

2. **RFC-0021 (wolfharness)** — Agent Concurrent Execution Safety
   Location: `packages/wolfharness/docs/rfcs/implemented/RFC-0021-agent-concurrent-execution-safety.md`

3. **BackgroundTaskProvider Implementation**
   Location: `packages/xeno-agent/src/xeno_agent/wolfharness/resource_providers/background_task_provider.py`

4. **Research Findings**
   Location: `.omo/notepads/background-task-provider/learnings.md`

5. **Decision Log**
   Location: `.omo/notepads/background-task-provider/decisions.md`

### External Resources

1. **MCP SEP-1686** — Asynchronous Task Tools (Model Context Protocol)
2. **Pydantic-AI `cancel_and_drain`** — `packages/pydantic-ai/pydantic_ai_slim/pydantic_ai/_utils.py`
3. **Pydantic-AI Agent Loop** — `packages/pydantic-ai/pydantic_ai_slim/pydantic_ai/agent.py`

### Code Locations

| Component | File |
|-----------|------|
| xeno-agent BackgroundTaskManager | `packages/xeno-agent/src/xeno_agent/task/manager.py` |
| xeno-agent BackgroundTask types | `packages/xeno-agent/src/xeno_agent/task/types.py` |
| xeno-agent BackgroundTaskProvider | `packages/xeno-agent/src/xeno_agent/wolfharness/resource_providers/background_task_provider.py` |
| Pydantic-AI cancel_and_drain | `packages/pydantic-ai/pydantic_ai_slim/pydantic_ai/_utils.py:223-240` |
| AgentPool SessionController | `packages/wolfharness/src/wolfharness/sessions/controller.py` |
| AgentPool SessionConfig | `packages/wolfharness/src/wolfharness/sessions/config.py` |

---

**End of RFC-0034**
