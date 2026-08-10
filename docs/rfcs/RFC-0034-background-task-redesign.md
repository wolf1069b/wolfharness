---
rfc_id: RFC-0034
title: BackgroundTask Architecture Redesign for AgentPool
status: DRAFT
author: yuchen.liu
reviewers: []
created: 2026-06-09
last_updated: 2026-06-09
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

---

## 3. Problem Statement

### 3.1 Specific Problems

1. **UI Events != LLM Context**: `SubAgentEvent` emitted by background tasks reaches the UI stream via `ctx.events.emit_event()`, but the lead agent's LLM conversation context (its `message_history`) is never updated. The LLM is unaware of task completion unless the user explicitly calls `background_output`.

2. **Fire-and-forget notification bug**: The `_on_task_completed` callback in `BackgroundTaskProvider._task_async()` spawns `_notify_parent()` via `asyncio.create_task()` without `cancel_and_drain`. When `session_pool.inject_prompt()` raises (e.g., parent session ended), the exception is silently lost.

3. **Empty `parent_session_id`**: Line 379 of `background_task_provider.py` falls back to `getattr(ctx.node, 'session_id', '')` — an empty string prevents `SessionPool.inject_prompt()` from routing the notification correctly.

4. **No upstream abstraction**: Every downstream project that wants background tasks must copy xeno-agent's implementation. There is no wolfharness-native primitive.

5. **String-based API**: Returning raw string IDs makes it impossible for the LLM to introspect task state without making another tool call.

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

    Returns a JSON-serializable dict with status, duration, and metadata.
    """
    ...

async def task_result(
    ctx: AgentContext,
    task_id: str,
    timeout_seconds: float = 60.0,
) -> str:
    """Retrieve the result of a background task, optionally blocking.

    Replaces ``background_output(block=True)``.
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
