---
rfc_id: RFC-0029
title: Agent Reactivation via Pending Prompt Queue
status: DRAFT
author: yuchen.liu
reviewers: []
created: 2026-04-26
last_updated: 2026-04-27 (Rev 2 — PR review feedback)
decision_date:
related_rfcs:
  - RFC-0021 (Agent Concurrent Execution Safety — per-run context isolation)
  - RFC-0028 (Delegation Provider Session Adaptation — child session lifecycle)
---

# RFC-0029: Agent Reactivation via Pending Prompt Queue

## Overview

This RFC proposes adding a **pending prompt queue** with `asyncio.Event` notification to `BaseAgent`, enabling agents to receive and process prompts injected while idle (between `run_stream()` calls). Currently, `inject_prompt()` silently drops messages when no active run context exists — this breaks async background task completion callbacks where the lead agent has finished streaming before the background task completes.

The design follows a **notification pattern**: the agent provides the signal (queue + event), the caller (server or direct API user) provides the reactivation loop. `run_stream()` semantics remain unchanged — it starts, processes queued prompts, completes. The pending prompt queue is drained at the start of each new `run_stream()` call.

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

After RFC-0021, each `run_stream()` call creates an isolated `AgentRunContext` with its own `PromptInjectionManager`. The injection flow has two phases:

1. **Producer**: `inject_prompt()` / `queue_prompt()` deliver messages to the current run's `injection_manager` via the `_current_run_ctx → _active_run_ctx → _background_run_ctx` fallback chain (added in RFC-0028 fix commit `aba2f9cb5`).
2. **Consumer**: Hook managers (`NativeAgentHookManager`, `ClaudeCodeAgentHookManager`) and `MCPToolBridge` consume injections after tool calls, attaching them as `additional_context` to tool results.

The `run_stream()` loop (base_agent.py L610-725) processes queued prompts:

```python
WHILE injection_manager.has_queued() AND NOT cancelled:
    prompt = injection_manager.pop_queued()
    async for event in self._run_stream_once(prompt):
        yield event
    injection_manager.flush_pending_to_queue()
FINALLY:
    injection_manager.clear()
    _active_run_ctx = None  # ← Destroys reactivation capability
```

**When the queue empties, the loop exits.** `_active_run_ctx` is set to `None` in the `finally` block. Any subsequent `inject_prompt()` call finds all three fallback contexts (`_current_run_ctx`, `_active_run_ctx`, `_background_run_ctx`) as `None` and silently drops the message.

### Key Use Case: Async Background Task Completion

In the xeno-agent library's `BackgroundTaskProvider`:

1. Lead agent starts an async background task via `task(async_mode=True)`
2. Lead agent's current `run_stream()` iteration completes (queue empties, loop exits)
3. Background task finishes → `_on_task_completed()` callback fires on its own `asyncio.Task`
4. Callback calls `ctx.agent.inject_prompt(notice)` — **silently dropped** because `_active_run_ctx` is `None`

The sync subagent path works because the lead agent's tool call blocks until the subagent completes — `run_stream()` is still active with `_active_run_ctx` set.

### OpenCode (TypeScript) Reference

OpenCode solves this with a **Runner state machine**:

- 4-state FSM: Idle | Running | Shell | ShellThenRun
- `ensureRunning(work)`: Idle → start work as Fiber → Running; Running → await existing run's Deferred
- `while(true)` loop inside Fiber — breaks only when assistant finished AND no tool calls
- New user messages written to DB, running loop re-reads each iteration
- Server-level `_run_async_prompt_queue()` drains prompts and starts new `run_stream()` per prompt

### Glossary

| Term | Definition |
|------|------------|
| **Pending Prompt Queue** | Agent-level `asyncio.Queue[str]` that receives injected prompts when no active run context exists (mailbox pattern) |
| **Pending Prompt Event** | `asyncio.Event` that fires when a prompt is added to the pending queue |
| **Reactivation** | Starting a new `run_stream()` call to process prompts that arrived while the agent was idle |
| **Caller-driven reactivation** | The caller (server or direct user) provides the reactivation loop; the agent provides the notification |
| **Mailbox pattern** | Agent's pending queue for `inject_prompt()` notices — string-only, consumed by the agent at `run_stream()` startup |
| **Request routing** | Server-level queue for full user requests (e.g., `QueuedAsyncPrompt` with model/agent metadata) — separate from the agent mailbox |

---

## Problem Statement

### Specific Problem

`inject_prompt()` and `queue_prompt()` silently drop messages when the agent is idle (no active `run_stream()` call). This breaks the async background task completion use case and any other scenario where prompts arrive between run iterations.

### Evidence

1. **xeno-agent `BackgroundTaskProvider._on_task_completed()`** (background_task_provider.py L803-811): Calls `ctx.agent.inject_prompt(notice)` from a different async task. If the lead agent has finished streaming, the injection is silently dropped.

2. **OpenCode server `_run_async_prompt_queue()`** (message_routes.py L602): Implements its own prompt queue and reactivation loop at the server level, duplicating what should be agent-level capability.

3. **Manual testing**: Starting a background task, letting the lead agent finish, then checking if the completion notice is received — it is not.

### Impact of Not Solving

- **Async background tasks are unreliable**: Completion notices lost, lead agent never learns results
- **Server-level workarounds proliferate**: Each server implements its own reactivation, duplicating logic
- **API inconsistency**: `inject_prompt()` sometimes works (during run) and sometimes silently drops (idle) — caller can't distinguish

---

## Goals & Non-Goals

### Goals

1. **No silent drops**: `inject_prompt()` / `queue_prompt()` always deliver — either to active run or to pending queue
2. **Reactivation signal**: Agent provides notification mechanism for callers to detect pending prompts
3. **Backward compatible**: Existing `run_stream()` calls behave identically when no pending prompts exist
4. **Server consolidation**: OpenCode server can use `wait_for_pending_prompt()` as a reactivation signal, while retaining its own request queue for full user request routing
5. **Minimal change**: Preserve `run_stream()` semantics (starts, processes, completes) — no "awaiting forever" mode

### Non-Goals

1. **Self-reactivation**: Agent will NOT start its own `run_stream()` — the caller drives the run lifecycle
2. **Bounded queue / backpressure**: No upper limit on pending prompts in this RFC (follow-up if needed)
3. **Priority ordering**: Pending prompts are FIFO — no priority scheduling
4. **OpenAI API server reactivation**: HTTP request-response protocol can't hold connections for reactivation; background task results require polling/webhook (separate concern)
5. **Cross-process notification**: Pending prompts are in-process only — no IPC mechanism

---

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **No silent drops** | Critical | `inject_prompt()` never silently discards a message |
| **Backward compatibility** | High | Existing `run_stream()` calls unchanged when no pending prompts |
| **Server integration effort** | High | Minimal changes to existing server implementations |
| **Resource safety** | High | No task/context leaks; idle agents don't hold resources |
| **Implementation complexity** | Medium | Simpler is better; fewer moving parts |
| **Testability** | Medium | Behavior is easy to unit test without complex fixtures |

---

## Options Analysis

### Option A: Awaitable Prompt Source (asyncio.Queue inside run_stream)

**Description**: Replace `injection_manager.has_queued()` with `await prompt_queue.get()` inside `run_stream()`. When queue empty, agent awaits instead of exiting. New prompt/injection → queue → agent wakes and processes.

**Advantages**:
- Minimal structural change to `run_stream()` — just change the loop condition
- Agent self-reactivates — no server cooperation needed
- Natural asyncio.Queue semantics (blocking get, put from any task)

**Disadvantages**:
- `run_stream()` becomes a long-lived coroutine — may never return
- `_active_run_ctx` lifecycle changes — must remain alive while awaiting
- Requires idle timeout mechanism to prevent resource leaks
- Cancellation semantics complex — what happens to `run_stream()` when agent is "awaiting"?
- Breaks the current pattern where `run_stream()` is called per-prompt by servers
- Difficult to reason about cleanup: when does `finally` block run?

**Evaluation Against Criteria**:

| Criterion | Score | Notes |
|-----------|-------|-------|
| No silent drops | ✅ Good | Queue never drops |
| Backward compatibility | ❌ Poor | run_stream() never returns |
| Server integration effort | ❌ Poor | All servers need restructuring |
| Resource safety | ❌ Poor | Long-lived coroutines |
| Implementation complexity | ⚠️ Medium | Simple change, complex implications |
| Testability | ⚠️ Medium | Timeout-based tests fragile |

**Effort Estimate**: Medium (1-2 days) for core, but high risk of regressions across all servers

**Risk Assessment**: High — changing `run_stream()` from "process and return" to "await forever" is a fundamental semantic shift

---

### Option B: Runner State Machine (OpenCode Pattern)

**Description**: New `Runner` class with Idle/Running states and `ensure_running()` method. Agent has a `runner` property that manages the `run_stream()` lifecycle. `inject_prompt()` on idle agent → `runner.ensure_running()` → starts new `run_stream()`.

**Advantages**:
- Clean separation of concerns — Runner manages lifecycle, Agent manages processing
- Matches proven OpenCode pattern
- Atomic state transitions prevent race conditions
- Natural mapping: `ensure_running()` → start or await existing run

**Disadvantages**:
- Significant new class with state machine logic
- Agent cannot safely start its own `run_stream()` — needs caller to drive async iterator
- All 4 servers must adapt to Runner API
- `run_stream()` remains public API — Runner would wrap it, creating two entry points
- More complex than needed for the immediate problem

**Evaluation Against Criteria**:

| Criterion | Score | Notes |
|-----------|-------|-------|
| No silent drops | ✅ Good | Runner ensures delivery |
| Backward compatibility | ⚠️ Medium | New Runner class alongside run_stream |
| Server integration effort | ❌ Poor | All servers need Runner adaptation |
| Resource safety | ✅ Good | Runner manages lifecycle explicitly |
| Implementation complexity | ❌ Poor | New state machine class |
| Testability | ⚠️ Medium | State machine testing adds complexity |

**Effort Estimate**: Large (2-4 days) for Runner + all server adaptations

**Risk Assessment**: Medium — clean design but high integration surface

---

### Option C: Server-Level Only (Extend OpenCode Pattern to All Servers)

**Description**: Don't change agent internals. Each server implements its own reactivation (like OpenCode's `_run_async_prompt_queue()`). `inject_prompt()` on idle agent adds to a server-managed queue.

**Advantages**:
- No change to agent API or internals
- Each server adapts independently
- Existing OpenCode server already has this pattern

**Disadvantages**:
- **No agent-level solution** — `inject_prompt()` still silently drops on idle agents without server
- Duplicated logic across 4 servers
- Non-server callers (direct API users, xeno-agent standalone) have no reactivation
- No single source of truth for pending prompts

**Evaluation Against Criteria**:

| Criterion | Score | Notes |
|-----------|-------|-------|
| No silent drops | ❌ Poor | Only works with server cooperation |
| Backward compatibility | ✅ Good | No agent changes |
| Server integration effort | ⚠️ Medium | Each server independently |
| Resource safety | ✅ Good | No new agent-level resources |
| Implementation complexity | ⚠️ Medium | Duplicated across servers |
| Testability | ⚠️ Medium | Per-server testing |

**Effort Estimate**: Medium (1-2 days per server × 3 servers)

**Risk Assessment**: Medium — solves server case but leaves API gap

---

### Option D: Pending Prompt Queue + Event Notification (Recommended)

**Description**: Add `_pending_prompts: asyncio.Queue[str]` and `_pending_prompt_event: asyncio.Event` to `BaseAgent`. When `inject_prompt()` finds no active run context, queue to `_pending_prompts` and set the event. `run_stream()` drains `_pending_prompts` at startup. Caller (server) can `await agent.wait_for_pending_prompt()` to detect when reactivation is needed.

**Advantages**:
- **No silent drops** — prompts always go somewhere (active run OR pending queue)
- **Backward compatible** — existing `run_stream()` calls unchanged when no pending prompts
- **Single source of truth** — agent's `_pending_prompts` queue replaces server-level duplicates
- **Caller-driven reactivation** — agent provides signal, caller provides loop (matches established pattern)
- **`run_stream()` semantics preserved** — starts, processes, completes; no "awaiting forever"
- **`_active_run_ctx` lifecycle unchanged** — per-run creation/cleanup
- **Simple** — queue + event + drain-on-start; no new classes or state machines

**Disadvantages**:
- **Requires caller cooperation** — agent can't self-reactivate; caller must await event and start new `run_stream()`
- **Event management edge cases** — must clear event correctly, handle race between event set and queue drain
- **OpenAI API server excluded** — request-response protocol can't hold connections for reactivation

**Evaluation Against Criteria**:

| Criterion | Score | Notes |
|-----------|-------|-------|
| No silent drops | ✅ Good | Queue always accepts |
| Backward compatibility | ✅ Good | No change when no pending prompts |
| Server integration effort | ✅ Good | Minimal — await event + start run_stream |
| Resource safety | ✅ Good | Queue is bounded by memory; event is lightweight |
| Implementation complexity | ✅ Good | Simple queue + event + drain |
| Testability | ✅ Good | Easy to unit test with asyncio primitives |

**Effort Estimate**: Short (2-4h) for core mechanism + OpenCode server integration; Medium (1-2d) for all servers

**Risk Assessment**: Low — additive change with no semantic shifts

---

## Recommendation

**Option D: Pending Prompt Queue + Event Notification** is recommended.

It scores highest across all evaluation criteria by providing the minimum viable mechanism that solves the problem without changing `run_stream()` semantics. The key insight is that reactivation is the **caller's responsibility** — the agent provides the notification (event + queue), the caller provides the reactivation loop. This matches the established pattern where `run_stream()` is a coroutine called by the server.

The accepted trade-off is that agents cannot self-reactivate. This is acceptable because:
1. `run_stream()` is an async iterator — it needs a caller to `async for` over it
2. Starting a new `run_stream()` from inside the agent (while idle) has no natural caller
3. All current use cases have a server or direct API caller that can provide the reactivation loop

---

## Technical Design

### 1. New Instance Variables on `BaseAgent`

```python
class BaseAgent:
    def __init__(self, ...):
        # Existing
        self._active_run_ctx: AgentRunContext | None = None
        self._background_run_ctx: AgentRunContext | None = None

        # New: Pending prompt queue for idle-state injection
        self._pending_prompts: asyncio.Queue[str] = asyncio.Queue()
        self._pending_prompt_event: asyncio.Event = asyncio.Event()
```

### 2. Modified `inject_prompt()` and `queue_prompt()`

Terminal fallback queues to `_pending_prompts` instead of silently dropping:

```python
def inject_prompt(self, message: str) -> None:
    """Inject a prompt into the active run, or queue for next run_stream()."""
    run_ctx = self._current_run_ctx or self._active_run_ctx or self._background_run_ctx
    if run_ctx is not None:
        run_ctx.injection_manager.inject(message)
    else:
        # No active run — queue for next run_stream() call
        self._pending_prompts.put_nowait(message)
        self._pending_prompt_event.set()

def queue_prompt(self, message: str) -> None:
    """Queue a prompt for the active run, or queue for next run_stream()."""
    run_ctx = self._current_run_ctx or self._active_run_ctx or self._background_run_ctx
    if run_ctx is not None:
        run_ctx.injection_manager.queue(message)
    else:
        self._pending_prompts.put_nowait(message)
        self._pending_prompt_event.set()
```

### 3. New Public Methods

```python
async def wait_for_pending_prompt(self, timeout: float = 3600.0) -> str | None:
    """Wait for a pending prompt to arrive. Returns None on timeout or if stolen.

    Callers (servers) use this as a reactivation signal. The method
    waits for the event, then attempts a non-blocking get from the queue.
    If the queue was drained by another consumer (e.g., a concurrent
    run_stream startup) between the event firing and the get, returns None.

    NOTE: This method only signals that a prompt *was* available. The
    caller should treat it as a wakeup signal and then drain the queue
    separately, rather than relying on the returned prompt value for
    processing. This avoids the "signal-then-get" race condition where
    an item could be stolen between event.wait() and queue.get().
    """
    try:
        await asyncio.wait_for(self._pending_prompt_event.wait(), timeout=timeout)
    except TimeoutError:
        return None
    # Use get_nowait() instead of await get() to avoid blocking if
    # another consumer drained the queue between event.wait() and here.
    try:
        prompt = self._pending_prompts.get_nowait()
    except asyncio.QueueEmpty:
        # Queue was drained by another consumer (e.g., concurrent run_stream)
        return None
    # Re-set event if more prompts remain
    if not self._pending_prompts.empty():
        self._pending_prompt_event.set()
    else:
        self._pending_prompt_event.clear()
    return prompt

def has_pending_prompts(self) -> bool:
    """Check if there are pending prompts waiting for the next run_stream()."""
    return not self._pending_prompts.empty()

def pop_pending_prompt(self) -> str | None:
    """Pop a pending prompt without waiting. Returns None if empty."""
    try:
        prompt = self._pending_prompts.get_nowait()
    except asyncio.QueueEmpty:
        return None
    if self._pending_prompts.empty():
        self._pending_prompt_event.clear()
    return prompt

def clear_pending_prompts(self) -> None:
    """Clear all pending prompts."""
    while not self._pending_prompts.empty():
        with contextlib.suppress(asyncio.QueueEmpty):
            self._pending_prompts.get_nowait()
    self._pending_prompt_event.clear()
```

### 4. Modified `run_stream()` Startup

Drain pending prompts into the new run's `injection_manager` before entering the while loop:

```python
async def run_stream(self, *prompts, ...):
    run_ctx = AgentRunContext(...)
    self._active_run_ctx = run_ctx
    token = _current_run_ctx_var.set(run_ctx)

    # NEW: Drain any pending prompts from idle state
    # NOTE: This drain loop and the event.clear() below are synchronous
    # (no await between them), so in single-threaded asyncio there is no
    # race condition with inject_prompt() — it cannot interleave between
    # the drain loop and the clear() call.
    while True:
        try:
            pending = self._pending_prompts.get_nowait()
        except asyncio.QueueEmpty:
            break
        run_ctx.injection_manager.queue(pending)

    # Clear event since we're draining
    self._pending_prompt_event.clear()

    # Existing while loop
    while run_ctx.injection_manager.has_queued() and not self._cancelled:
        ...
```

### 5. Modified `has_queued_prompts()` and Related Query Methods

Add `_pending_prompts` check to the existing fallback chain:

```python
def has_queued_prompts(self) -> bool:
    """Check if there are queued prompts in the active run or pending queue."""
    run_ctx = self._current_run_ctx or self._active_run_ctx or self._background_run_ctx
    if run_ctx is not None:
        return run_ctx.injection_manager.has_queued()
    return not self._pending_prompts.empty()

def has_pending_injections(self) -> bool:
    """Check if there are pending injections in the active run or pending queue."""
    run_ctx = self._current_run_ctx or self._active_run_ctx or self._background_run_ctx
    if run_ctx is not None:
        return run_ctx.injection_manager.has_pending()
    return not self._pending_prompts.empty()
```

### 6. OpenCode Server Integration

The agent's pending prompt queue serves as a **mailbox** for `inject_prompt()` notices (string-only). The OpenCode server retains its own `QueuedAsyncPrompt` queue for full user request routing (which includes model selection, agent name, and message metadata that cannot be reduced to a string).

The two queues serve different concerns:
- **Agent mailbox** (`_pending_prompts`): Async notifications from background tasks, system events, etc.
- **Server request queue** (`pending_async_prompts`): Full user requests with routing metadata

`wait_for_pending_prompt()` acts as a **reactivation signal** — the server awaits it to know when something needs processing, then checks both queues:

```python
# REVISED: Agent mailbox as signal, server retains its own request queue
async def _run_async_prompt_queue(
    self,
    session_id: str,
    state: ServerState,
) -> None:
    """Drain both server request queue and agent mailbox.

    Drain ordering: server requests first, then agent mailbox.
    This ensures user-typed messages always preempt background task
    notices. Be aware this creates a fairness implication — a flood
    of user messages could starve background task notices. This is
    intentional: user-facing responsiveness takes priority.
    """
    while True:
        # 1. Process any pending server requests first (full metadata)
        while state.pending_async_prompts:
            queued = state.pending_async_prompts.pop(0)
            await self._process_message_locked(
                session_id=session_id,
                request=queued.request,
                state=state,
                user_msg_id=queued.user_msg_id,
                user_msg_with_parts=queued.user_msg_with_parts,
            )

        # 2. Drain agent mailbox for injected prompts
        #    Use pop_pending_prompt() directly to avoid TOCTOU race
        #    between has_pending_prompts() and pop.
        agent = state.sessions.get(session_id)
        if agent is not None:
            while True:
                prompt = agent.pop_pending_prompt()  # Returns None if empty
                if prompt is None:
                    break
                # Adapt string notice to MessageRequest for _process_message_locked.
                request = MessageRequest.from_injected_prompt(prompt)
                await self._process_message_locked(
                    session_id=session_id,
                    request=request,
                    state=state,
                    user_msg_id=None,
                    user_msg_with_parts=None,
                )

        # 3. Wait for next signal from agent mailbox.
        #    wait_for_pending_prompt() is used as a WAKEUP SIGNAL only —
        #    its return value may be None (e.g., if another consumer
        #    drained the queue). The actual prompt consumption happens
        #    in step 2 above via pop_pending_prompt().
        agent = state.sessions.get(session_id)
        if agent is not None:
            result = await agent.wait_for_pending_prompt(timeout=3600)
            if result is None:
                break  # timeout — no more activity
            # result may contain a prompt, but we loop back to steps 1+2
            # which drain both queues properly. Don't process result here
            # to avoid duplicate processing or missing the server queue.
        else:
            break
```

This preserves the server's `QueuedAsyncPrompt` queue for full request routing while using the agent's mailbox as a lightweight reactivation signal. A `MessageRequest.from_injected_prompt()` factory method (or equivalent adapter) is required to bridge the string mailbox content to the server's `MessageRequest`-based API.

### 7. ACP / AG-UI Server Integration

Add reactivation loops for long-lived connections:

```python
# ACP server streaming handler
async def handle_acp_streaming(self, agent, initial_prompt):
    # First run
    async for event in agent.run_stream(initial_prompt):
        yield convert_event(event)

    # Reactivation loop
    while True:
        prompt = await agent.wait_for_pending_prompt(timeout=1800)
        if prompt is None:
            break
        async for event in agent.run_stream(prompt):
            yield convert_event(event)
```

### 8. Event Ordering and Race Conditions

**Race: `inject_prompt()` sets event, then `wait_for_pending_prompt()` clears it before draining**

Solution: `wait_for_pending_prompt()` clears event AFTER draining one prompt, then re-sets if queue is non-empty. The event is a "at least one prompt available" signal, not a "exactly one prompt" signal.

**Race: `run_stream()` drains queue while `inject_prompt()` adds to it**

Solution: `asyncio.Queue` is thread-safe for single-threaded async code. `put_nowait()` and `get_nowait()` are atomic within the event loop. The drain loop in `run_stream()` runs synchronously before yielding control, so no interleaving with `inject_prompt()` calls.

**Race: Two callers both `await wait_for_pending_prompt()`**

Solution: First caller gets the prompt (queue.get() is atomic). Second caller sees empty queue and waits again. This is correct — prompts are delivered exactly once.

### 9. Resource Safety

- `_pending_prompts` queue is unbounded (memory only). Acceptable for typical use cases where prompt volume is low (background task completions, user messages).
- `_pending_prompt_event` is lightweight — no task allocation, no timer.
- No long-lived `asyncio.Task` is created by the agent — the caller decides when to start `run_stream()`.
- `clear_pending_prompts()` provided for cleanup during agent shutdown. Should be wired into `__aexit__` or `stop()` (Phase 4).
- **Event loop affinity**: `asyncio.Queue` and `asyncio.Event` are not thread-safe. All callers must be in the same event loop. This matches current wolfharness architecture where all agent operations are asyncio-based.

---

## Security Considerations

1. **Injection source validation**: `inject_prompt()` does not validate the source of injected messages. Callers should ensure only trusted code can call `inject_prompt()` on an agent.
2. **Queue overflow**: Unbounded queue could be exploited by a malicious caller flooding `inject_prompt()`. Consider adding a configurable max queue size in a follow-up.
3. **Cross-session leakage**: `_pending_prompts` is per-agent-instance, not per-session. If an agent is reused across sessions (after RFC-0026, this should be rare), prompts from one session could leak into another. The `run_stream()` drain happens at the start, so any stale prompts from a previous session would be processed first.

---

## Implementation Plan

### Phase 1: Core Mechanism (2-4h)

1. Add `_pending_prompts` and `_pending_prompt_event` to `BaseAgent.__init__`
2. Modify `inject_prompt()` and `queue_prompt()` to use pending queue as terminal fallback
3. Modify `run_stream()` to drain pending prompts at startup
4. Add `wait_for_pending_prompt()`, `has_pending_prompts()`, `pop_pending_prompt()`, `clear_pending_prompts()`
5. Modify `has_queued_prompts()` and `has_pending_injections()` to check pending queue
6. Write unit tests in `test_inject_prompt_cross_task.py`

### Phase 1.5: Continuous Mode Injection Consumption (1-2h)

1. Modify `_continuous()` loop (base_agent.py L472-504) to drain `self._background_run_ctx.injection_manager` at the start of each iteration (note: continuous mode uses `_background_run_ctx`, not `_active_run_ctx`)
2. This ensures `inject_prompt()` delivered to `_background_run_ctx.injection_manager` during continuous mode is actually consumed
3. Write unit test for continuous mode injection consumption

### Phase 2: OpenCode Server Integration (2-4h)

1. Modify `_run_async_prompt_queue()` to use `wait_for_pending_prompt()` as reactivation signal
2. Add `MessageRequest.from_injected_prompt()` factory method (or equivalent adapter) to wrap injected string prompts as `MessageRequest` for `_process_message_locked()`
3. Retain `ServerState.pending_async_prompts` for full user request routing (mailbox vs request routing separation)
4. Use `pop_pending_prompt()` directly instead of `has_pending_prompts()` + `pop` to avoid TOCTOU race
5. Test integration with OpenCode server test fixtures

### Phase 3: ACP/AG-UI Server Integration (4-8h)

1. Add reactivation loops to ACP streaming handler
2. Add reactivation loops to AG-UI event handler
3. Test with ACP and AG-UI test fixtures

### Phase 4: Documentation and Cleanup (2-4h)

1. Update docstrings for modified methods
2. Add pending prompt behavior to agent usage guide
3. Wire `clear_pending_prompts()` into `__aexit__` or `stop()` for clean shutdown
4. Verify all existing tests still pass

---

## Open Questions

1. **Should `_pending_prompts` have a configurable max size?** Current design is unbounded. If a background task floods `inject_prompt()`, memory could grow. A `max_pending_prompts` parameter with drop-oldest or reject-new behavior could be added.

2. **Should `wait_for_pending_prompt()` support cancellation?** The current design uses `asyncio.wait_for()` which raises `TimeoutError` on timeout. Should we also support `CancelledError` propagation? (Likely yes — follow asyncio conventions.)

3. **Should pending prompts be persisted?** If the process restarts while prompts are in the pending queue, they are lost. For production reliability, should prompts be written to `SessionStore`? (Likely no for this RFC — follow-up if needed.)

4. **How does this interact with `_background_run_ctx`?** The existing fallback chain is `_current_run_ctx → _active_run_ctx → _background_run_ctx → _pending_prompts`. If an agent has `run_in_background()` active, prompts go to `_background_run_ctx` (not pending queue). **However**, the `_continuous()` loop (base_agent.py L472-504) does NOT currently consume from `injection_manager` — it only processes the initial `prompt` argument. This means injections delivered to `_background_run_ctx.injection_manager` during continuous mode will be stuck and never consumed. **Resolution**: The `_continuous()` loop should drain `injection_manager` at the start of each iteration (similar to the proposed `run_stream()` change). This is added to the implementation plan as Phase 1.5.

5. **Should `inject_prompt()` return a boolean indicating delivery path?** This would help callers distinguish "delivered to active run" vs "queued for later". Could be useful for logging but adds API surface.

6. **Multi-protocol event contention**: If the same agent is exposed via multiple servers (e.g., ACP + OpenCode simultaneously), both could `await wait_for_pending_prompt()` on the same agent. The first caller wins the prompt — the second gets nothing. Is this acceptable, or should we add per-server event dispatching? (Likely acceptable for now — multi-protocol exposure is uncommon.)

---

## Decision Record

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-26 | Option D (Pending Prompt Queue) selected | Minimal change, backward compatible, solves core problem |
| 2026-04-26 | Caller-driven reactivation chosen over self-reactivation | Agent can't safely start its own run_stream() — needs caller for async iteration |
| 2026-04-26 | Unbounded queue accepted for initial implementation | Typical use case is low-volume; bounded queue can be added later |
| 2026-04-27 | Agent mailbox vs server request routing separation (PR review) | Agent's `_pending_prompts` is string-only mailbox for `inject_prompt()` notices; server retains `QueuedAsyncPrompt` queue for full user request routing with model/agent metadata |
| 2026-04-27 | `_continuous()` loop needs injection consumption (PR review) | `_continuous()` loop does not consume `injection_manager`, causing injections delivered to `_background_run_ctx` to be stuck; added Phase 1.5 to implementation plan |
| 2026-04-27 | `MessageRequest.from_injected_prompt()` adapter required (Oracle review) | `_process_message_locked` expects `MessageRequest`, not `str`; server integration needs adaptation layer to wrap injected string prompts |
| 2026-04-27 | Server request drain ordering: user requests preempt mailbox (Oracle review) | Intentional design for user-facing responsiveness; documented fairness implication |
| 2026-04-27 | `wait_for_pending_prompt()` uses `get_nowait()` instead of `await get()` (PR review) | Avoids "signal-then-get" race where another consumer could drain the queue between `event.wait()` and `queue.get()`, causing indefinite blocking |
| 2026-04-27 | `wait_for_pending_prompt()` is wakeup signal only, not prompt consumer (PR review) | Server integration uses it as a trigger to loop back and drain both queues via `pop_pending_prompt()`, avoiding lost prompts from the previous "signal returns prompt" pattern |

---

## References

- `src/wolfharness/agents/base_agent.py` — `inject_prompt()`, `queue_prompt()`, `run_stream()`
- `src/wolfharness/agents/prompt_injection.py` — `PromptInjectionManager`
- `src/wolfharness/agents/native_agent/hook_manager.py` — injection consumption in post-tool hooks
- `src/wolfharness_server/opencode_server/routes/message_routes.py` — `_run_async_prompt_queue()`
- xeno-agent `background_task_provider.py` L803-811 — `_on_task_completed()` callback
- OpenCode TypeScript: `effect/runner.ts` — Runner state machine
- OpenCode TypeScript: `session/prompt.ts` — `while(true)` run loop
- RFC-0021 — Per-run context isolation (AgentRunContext)
- RFC-0028 — Delegation provider session adaptation (child session lifecycle)
