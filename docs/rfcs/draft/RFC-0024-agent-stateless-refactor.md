---
rfc_id: RFC-0024
title: "Agent Stateless Refactor — Decouple Session State from BaseAgent"
status: DRAFT
author: yuchen.liu
reviewers: []
created: 2026-04-20
last_updated: 2026-04-20
decision_date:
related_rfcs:
  - RFC-0021 (Agent Concurrent Execution Safety)
  - RFC-0026 (Per-Session Agent Instances — Phase 1)
  - RFC-0025 (Shared Agent Architecture — Phase 3)
---

# RFC-0024: Agent Stateless Refactor — Decouple Session State from BaseAgent

> **Phase 2 of the Multi-Session Isolation Roadmap.** Depends on [RFC-0026](../implemented/RFC-0026-per-session-agent-isolation.md) (Phase 1). Enables [RFC-0025](./RFC-0025-shared-agent-architecture.md) (Phase 3).

## Overview

This RFC proposes refactoring `BaseAgent` to be stateless with respect to sessions: move session-scoped state (`conversation`, `_input_provider`, `session_id`) off the agent instance and into parameters passed at call time. This aligns wolfharness's agent model with pydantic-ai's design (where the agent is a stateless worker and conversation persistence is the caller's responsibility), and enables Phase 3's shared-agent architecture where a single `BaseAgent` instance serves all sessions.

## Background & Context

### pydantic-ai's Stateless Agent Design

pydantic-ai's `Agent` is designed to be **stateless**:

```python
# pydantic-ai official pattern:
agent = Agent("openai:gpt-4o")

# Conversation persistence is the CALLER's responsibility
result1 = await agent.run("hello")
result2 = await agent.run("continue", message_history=result1.new_messages())
# `message_history` is a parameter, not agent instance state
```

Key design elements:
1. **`message_history`** — A separate parameter on `run()`/`run_stream()`, NOT an instance variable
2. **`deps`** — Caller-injected business dependencies, per-call, NOT per-agent
3. **`GraphAgentState`** — Internal framework state, created per-run, returned via `AgentRunResult`, NOT stored on the agent

### wolfharness's Current Misalignment

`BaseAgent` holds session-scoped state as **instance variables**, making the agent stateful:

| Instance Variable | pydantic-ai Equivalent | Current Location | Should Be |
|---|---|---|---|
| `self.conversation` | `message_history` parameter | Instance variable | `run_stream(message_history=...)` parameter |
| `self._input_provider` | `deps` or parameter | Instance variable | `run_stream(input_provider=...)` parameter |
| `self.session_id` | Per-run parameter | Instance variable | `run_stream(session_id=...)` parameter (already exists) |
| `self._active_run_ctx` | `GraphAgentState` (per-run) | Instance variable | `ContextVar` + session registry |
| `self._current_stream_task` | Per-run | Instance variable | `AgentRunContext.current_task` (already via RFC-0021) |
| `self._cancelled` | Per-run | Instance variable | `AgentRunContext.cancelled` (already via RFC-0021) |

### Why This Matters Now

After [RFC-0026](../implemented/RFC-0026-per-session-agent-isolation.md), each session has its own `BaseAgent` instance. This works, but has a cost: each instance spawns its own MCP subprocesses (~10–50MB, ~1–4s init). If `BaseAgent` were stateless, a single agent instance could serve all sessions (Phase 3), reducing resource cost by N× (where N = concurrent sessions).

## Problem Statement

### The Problem

`BaseAgent` mixes two categories of state on a single instance:

1. **Agent configuration** (immutable after init): model, tools, system prompts, MCP config — safe to share
2. **Session runtime state** (mutated per-run): conversation, session_id, input_provider, active_run_ctx — NOT safe to share

This coupling forces each session to have its own agent instance, incurring MCP subprocess cost per session.

### Impact

- **Resource waste**: 5 concurrent sessions × 2 MCP servers = 10 subprocesses, ~50–250MB
- **Architecture fragility**: New session-scoped state added to `BaseAgent` automatically becomes shared-state risk
- **pydantic-ai misalignment**: wolfharness's agent model diverges from the framework it's built on

## Goals & Non-Goals

### Goals

1. Move session-scoped state off `BaseAgent` instance and into `run_stream()` parameters or `AgentRunContext`
2. Make `BaseAgent` safe to share across sessions (no instance mutation during `run_stream()`)
3. Align with pydantic-ai's stateless agent model
4. Enable [RFC-0025](./RFC-0025-shared-agent-architecture.md) (shared-agent architecture)

### Non-Goals

1. **Not**: Implementing shared-agent architecture (that's Phase 3)
2. **Not**: Changing the server's session management (server still uses per-session agents after this RFC)
3. **Not**: Refactoring `AgentRunContext` internals (already provides per-run isolation via RFC-0021)
4. **Not**: Changing ACP/AG-UI/OpenAI API server implementations (follow-up)

### Success Criteria

- [ ] `BaseAgent` has zero session-scoped instance variables mutated during `run_stream()`
- [ ] `self.conversation` is not referenced in `_run_stream_once()` (replaced by `message_history` parameter)
- [ ] `self._input_provider` is not mutated by server code (replaced by `input_provider` parameter)
- [ ] `self.session_id` is not set as an instance variable during `run_stream()`
- [ ] `interrupt()` works correctly for concurrent sessions on the same agent (via `AgentRunContext`)
- [ ] All existing tests pass

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Stateless Correctness | Critical | No instance mutation during `run_stream()` |
| `interrupt()` Safety | Critical | `interrupt()` cancels the correct run, not cross-session |
| pydantic-ai Alignment | High | Agent model matches pydantic-ai's stateless design |
| Migration Complexity | Medium | Effort to update 24 `self.conversation` references + 10 `bind_agent_to_session` sites |
| Backward Compatibility | High | Existing call sites continue to work |

## Options Analysis

### Option 1: Full Stateless Refactor (Recommended)

Move all session-scoped state off `BaseAgent` into parameters and `AgentRunContext`:

1. `self.conversation` → `run_stream(message_history=...)` parameter (always passed)
2. `self._input_provider` → `run_stream(input_provider=...)` parameter (already accepted but not used by server)
3. `self.session_id` → `run_stream(session_id=...)` parameter (already passed, remove instance assignment)
4. `self._active_run_ctx` → `AgentRunContext` + session→run_ctx registry for `interrupt()`
5. `self._current_stream_task` → `AgentRunContext.current_task` (already isolated via RFC-0021)
6. `self._cancelled` → `AgentRunContext.cancelled` (already isolated via RFC-0021)

**Advantages**:
- Complete alignment with pydantic-ai's stateless model
- Enables shared-agent architecture (Phase 3)
- All session-scoped state is explicit at call sites
- Future session state additions are naturally scoped to parameters

**Disadvantages**:
- **`_active_run_ctx` requires a session registry**: `interrupt()` runs in a different async task than `run_stream()`. `ContextVar` returns `None` outside the originating task. A `session_id → AgentRunContext` mapping is needed for `interrupt()` to find the correct context.
- **24 `self.conversation` references** must be replaced in `_run_stream_once()` and related methods
- **`_internal_fs`** and **`staged_content`** are session-scoped but not easily parameterizable — they'd need to move to `AgentRunContext` or be passed as deps
- Largest change surface of the three phases

### Option 2: Partial Stateless Refactor (Conversation Only)

Move only `self.conversation` to `message_history` parameter. Keep `_input_provider`, `session_id`, `_active_run_ctx` as instance variables.

**Advantages**:
- Smaller change surface — only conversation-related refs
- Conversation is the largest source of shared-state issues (cleared in `create_session`, read/written in `_run_stream_once`)
- `_input_provider` is set once per session and not mutated during a run

**Disadvantages**:
- Agent is still not truly stateless — `_active_run_ctx`, `_input_provider`, `session_id` remain as instance variables
- Does NOT enable shared-agent architecture (Phase 3 still blocked)
- Partial alignment with pydantic-ai (conversation is the main alignment, but other state remains)

### Comparison

| Criterion | Option 1: Full Stateless | Option 2: Partial |
|-----------|-------------------------|-------------------|
| Stateless Correctness | ✅ Complete | ⚠️ Partial |
| `interrupt()` Safety | ✅ Session registry | ❌ Still instance-scoped |
| pydantic-ai Alignment | ✅ Full | ⚠️ Conversation only |
| Enables Phase 3 | ✅ Yes | ❌ No |
| Migration Complexity | High (~5 days) | Medium (~2 days) |

## Recommendation

**Option 1: Full Stateless Refactor.**

Option 2 doesn't enable Phase 3, which is the entire point of making the agent stateless. If we're going to refactor, we should do it completely. The effort difference (~3 extra days) pays for itself by enabling Phase 3's resource savings (eliminating N-1 agent instances × MCP subprocesses).

### Accepted Trade-offs

1. **Session→run_ctx registry for `interrupt()`**: This is unavoidable — `ContextVar` returns `None` outside the originating task. The registry is a simple `dict[str, AgentRunContext]` maintained by the agent itself (set in `run_stream()`, cleared on run completion). Not significantly more complex than the current `_active_run_ctx` instance variable.
2. **`_internal_fs` and `staged_content`**: These are session-scoped but not easily parameterizable. They should move to `AgentRunContext.deps` or `AgentRunContext` directly. This is a smaller change than conversation.
3. **Larger change surface**: 24 conversation refs + 10 bind_agent_to_session sites. Acceptable because the changes are mechanical (find-and-replace pattern).

## Technical Design

### State Migration Plan

#### 1. `self.conversation` → `message_history` parameter

`run_stream()` already accepts `message_history: list[ChatMessage] | None = None`. Currently, `_run_stream_once()` uses `message_history if message_history is not None else self.conversation` (line 766). The refactored path:

```python
# BEFORE (in _run_stream_once):
history = message_history if message_history is not None else self.conversation

# AFTER: message_history is ALWAYS passed by the server
history = message_history  # Required parameter, no fallback
```

**Server-side change** (all `run_stream` call sites):
```python
# BEFORE:
iterator = agent.run_stream(*user_prompt, session_id=session_id)

# AFTER:
iterator = agent.run_stream(
    *user_prompt,
    session_id=session_id,
    message_history=state.messages[session_id],
)
```

**Conversation mutation handling** — currently the server mutates `agent.conversation` in 4 places:

| Location | Mutation | After Refactor |
|----------|----------|---------------|
| `session_routes.py:694` | `agent.conversation.chat_messages.clear()` | Remove — new agent has empty conversation |
| `message_routes.py:536` | `agent.conversation.add_message(msg)` | `state.messages[session_id].append(msg)` |
| `session_routes.py:1437` | `agent.conversation.compact()` | `state.compact_messages(session_id)` |
| `session_routes.py:564,599` | `agent.conversation.chat_messages` (read) | `state.messages[session_id]` (read) |

**After refactor, `self.conversation` is REMOVED from `BaseAgent`** (hard cut — no dual-path deprecation). All callers must pass `message_history=` explicitly to `run_stream()`. This includes CLI usage (`wolfharness run`), which must manage its own message history list.

#### 1a. Non-Mechanical `self.conversation` Edge Cases

Not all 24 `self.conversation` references are simple find-and-replace:

| Edge Case | Location | Notes |
|---|---|---|
| `conversation._config` | `_run_stream_once()` | Config (compaction settings, etc.) must be available without `self.conversation`. Extract to `AgentRunContext` or agent constructor param. |
| `get_initialization_tasks()` | Agent init | References `self.conversation` during agent setup. Must use `message_history` parameter or accept empty list. |
| `as_tool` wrapper | Agent's `as_tool` method | Swaps conversation with a wrapped version. Must work with `message_history` parameter — swap the parameter, not the instance variable. |

### `load_session()` API Change

Currently, `load_session()` mutates `self.conversation.chat_messages` in place:

```python
# Current:
self.conversation.chat_messages.clear()
self.conversation.chat_messages.extend(loaded_messages)
```

After refactor, `load_session()` must **return** messages instead of mutating agent state:

```python
def load_session(self, session_id: str) -> list[ChatMessage]:
    """Load session messages from storage.

    Returns the message list for the caller to pass as `message_history=`
    to the next `run_stream()` call.
    """
    return self._storage.load_messages(session_id)
```

The server calls:
```python
messages = agent.load_session(session_id)
iterator = agent.run_stream(..., message_history=messages)
```

#### 2. `self._input_provider` → `input_provider` parameter

`run_stream()` already accepts `input_provider: InputProvider | None = None`. Currently, `get_context()` uses `input_provider or self._input_provider` (line 395). The refactored path:

```python
# BEFORE (in get_context):
provider = input_provider or self._input_provider

# AFTER: input_provider is ALWAYS passed by the server
provider = input_provider  # Required for server usage
```

**Server-side change**:
```python
# BEFORE:
agent._input_provider = state.ensure_input_provider(session_id)  # bind_agent_to_session

# AFTER:
iterator = agent.run_stream(
    *user_prompt,
    session_id=session_id,
    message_history=state.messages[session_id],
    input_provider=state.ensure_input_provider(session_id),
)
```

#### 3. `self.session_id` → `session_id` parameter only

`run_stream()` already accepts `session_id: str | None = None`. Currently, `run_stream()` sets `self.session_id = session_id` at line 644/652. Remove the instance assignment:

```python
# BEFORE (in run_stream):
self.session_id = session_id  # Instance mutation!

# AFTER: session_id is passed through AgentRunContext only
# self.session_id is NOT set as instance variable
run_ctx = AgentRunContext(session_id=session_id, ...)
```

**`ChatMessage.user_prompt` reads `self.session_id`** at line 774 — must be changed to read from `AgentRunContext`:

```python
# BEFORE:
session_id=self.session_id

# AFTER:
session_id=AgentRunContext.current().session_id
```

#### 4. `self._active_run_ctx` → Session Registry

The most complex change. `interrupt()` needs to find the run context for a given session from a different async task.

```python
# Added to BaseAgent (per-instance):
_active_runs: dict[str, AgentRunContext] = field(default_factory=dict)  # session_id → run_ctx

# In run_stream():
self._active_runs[session_id] = run_ctx
try:
    async for event in self._run_stream_once(run_ctx, ...):
        yield event
finally:
    self._active_runs.pop(session_id, None)
    # Stale entry cleanup: if generator cleanup doesn't execute finally,
    # entries may accumulate. Add TTL-based fallback in cleanup task.

# In interrupt():
async def interrupt(self, session_id: str, ...):
    run_ctx = self._active_runs.get(session_id)
    if run_ctx:
        run_ctx.current_task.cancel()
```

This replaces `self._active_run_ctx` with a per-instance registry keyed by `session_id`. The registry is safe for concurrent access within a single asyncio event loop (no lock needed — dict operations are atomic under GIL).

**Why per-instance dict, NOT ClassVar**: A ClassVar dict would be shared across all agent types — `ACPAgent` and `AGUIAgent` would share the same `_active_runs` dict, causing cross-agent-type collision where `ACPAgent.interrupt("session-1")` could accidentally cancel an `AGUIAgent`'s run. Per-instance dict ensures each agent instance has its own registry.

#### 4a. `AgentRunContext.session_id` Fix

Currently, `AgentRunContext.session_id` is initialized as `uuid.uuid4().hex` (random UUID), NOT connected to the `session_id` parameter passed to `run_stream()`. This must be fixed:

```python
# Current (base_agent.py ~line 656):
run_ctx = AgentRunContext(deps=deps)  # session_id = random UUID

# Fixed:
run_ctx = AgentRunContext(deps=deps, session_id=session_id)  # Uses caller's session_id
```

This ensures `AgentRunContext.current().session_id` returns the actual session ID, not a random value.

#### 5. `self._internal_fs` and `self.staged_content`

Move to `AgentRunContext`:

```python
@dataclass
class AgentRunContext:
    # ... existing fields ...
    internal_fs: IsolatedMemoryFileSystem
    staged_content: StagedContent
```

Created per-run from the session's state. The server provides these via a `SessionState` container:

```python
@dataclass
class SessionState:
    messages: list[ChatMessage]
    input_provider: InputProvider
    internal_fs: IsolatedMemoryFileSystem
    staged_content: StagedContent
```

**Migration note**: `AgentContext.internal_fs` currently delegates to `self.agent.internal_fs` (line reference: `running/context.py`). After moving `internal_fs` to `AgentRunContext`, `AgentContext.internal_fs` must delegate to `self._run_ctx.internal_fs` instead. This is a non-mechanical change — verify all `ctx.internal_fs` call sites.

### Agent Types Requiring Migration

Phase 2 must update ALL agent types, not just `BaseAgent`/`Agent`. The following agent types also reference `self.conversation.chat_messages`:

| Agent Type | File | References |
|---|---|---|
| ACPAgent | `agents/acp_agent.py:748-749` | `self.conversation.chat_messages.clear()` |
| ClaudeCodeAgent | `agents/claude_code_agent.py:1467-1468` | `self.conversation.chat_messages.clear()` / `.extend()` |
| CodexAgent | `agents/codex_agent.py:271-272, 689-690` | `self.conversation.chat_messages.clear()` / `.extend()` |
| AGUIAgent | `agents/agui_agent.py` | Similar patterns |

Each agent type's `interrupt()` method must also be updated to use `self._active_runs[session_id]` instead of `self._active_run_ctx`.

### Impact on `_run_stream_once()`

The 24 `self.conversation` references in `_run_stream_once()` and related methods become `run_ctx.message_history` or equivalent. This is the largest mechanical change.

### Impact on Tools

Tools that access `ctx.agent.conversation` or `ctx.agent._internal_fs` need to read from `AgentRunContext` instead. The `AgentContext` facade should be updated to redirect these accesses:

```python
class AgentContext:
    @property
    def conversation(self) -> list[ChatMessage]:
        """Access current run's message history."""
        return self._run_ctx.message_history
```

## Implementation Plan

### Duration: 4–6 days

| Phase | Scope | Duration |
|-------|-------|----------|
| **P2.1** | `message_history` parameter: replace 24 `self.conversation` refs in `_run_stream_once()`, update 7 server call sites, handle 4 mutation points | 2 days |
| **P2.2** | `input_provider` parameter: update `get_context()`, remove `bind_agent_to_session()`, update 10 server call sites | 1 day |
| **P2.3** | `session_id` parameter: remove instance assignment, update `ChatMessage.user_prompt`, add session registry for `_active_run_ctx`, update `interrupt()` | 1–2 days |
| **P2.4** | Move `_internal_fs` and `staged_content` to `AgentRunContext`, update `AgentContext` facade | 1 day |

### Dependencies

- Requires [RFC-0026](../implemented/RFC-0026-per-session-agent-isolation.md) (Phase 1) to be complete
- `AgentRunContext` (from RFC-0021) provides per-run isolation for event_queue, injection_manager, cancellation

### Rollback

Each phase is independently revertable. If `_active_run_ctx` registry proves problematic, `self._active_run_ctx` can be kept as instance variable (with per-session agent from Phase 1, this is safe).

## Open Questions

1. **`_active_runs` scope**: Should the session→run_ctx registry be class-level on `BaseAgent` or on a separate `RunRegistry`?
    - Context: Class-level dict is simple but couples all agent instances. A separate registry is cleaner but adds indirection.
    - Status: **Resolved** — per-instance dict, NOT ClassVar (avoids cross-agent-type collision as identified by Oracle/Metis review). ClassVar would cause `ACPAgent` and `AGUIAgent` to share the same dict, leading to `interrupt()` canceling the wrong agent type's run.

2. **Backward compatibility for `self.conversation`**: Should `BaseAgent.conversation` still exist as a fallback for non-server callers?
    - Context: Direct agent usage (e.g., `wolfharness run`) doesn't have a server managing message history.
    - Status: **Resolved** — hard cut, no dual-path. Remove `self.conversation` as fallback entirely. All callers must pass `message_history=` explicitly. Dual-path (deprecation warning) creates maintenance burden and masks migration errors, as identified by Metis review.

3. **`SessionState` dataclass**: Should the server consolidate per-session state into a single `SessionState` container?
    - Context: Currently spread across `state.messages[session_id]`, `state.input_providers[session_id]`, `state.agents[session_id]._internal_fs`. A `SessionState` would co-locate these.
    - Status: **Resolved** — consolidate into single `SessionState` dataclass. ServerState's 7+ per-session dicts (sessions, messages, session_locks, input_providers, agents, etc.) will be merged into `SessionState`.

4. **ACP/AG-UI/OpenAI API server migration**: These servers also use `self.conversation` and `bind_agent_to_session`. Should they be migrated in the same PR?
    - Context: ACP server already passes `input_provider=` to `run_stream()`. AG-UI and OpenAI API servers have simpler patterns.
    - Status: **Resolved** — ACPAgent, ClaudeCodeAgent, CodexAgent, AGUIAgent MUST be included in the migration PR (not separate PRs), because `interrupt()` on those agents will break if `_active_run_ctx` is removed from the base class.

## Decision Record

> Complete after RFC review.

---

## Review Notes

### Oracle + Metis Review (2026-04-20)

- **Phase 2 is the irreducible enabler for Phase 3** — without stateless refactoring, shared agent is unsafe
- **`_active_runs` must be per-instance dict, NOT ClassVar** — ClassVar causes cross-agent-type collision (ACPAgent, AGUIAgent share the same dict)
- **Hard cut for `self.conversation`** — no dual-path deprecation warning. Dual-path is maintenance burden and masks migration errors.
- **`AgentRunContext.session_id` is disconnected** — currently random UUID, must be connected to `run_stream(session_id=...)` parameter
- **`AgentContext.internal_fs` delegates to agent, not run_ctx** — migration path must update this delegation
- **All agent types need migration** — not just BaseAgent/Agent. ACPAgent, ClaudeCodeAgent, CodexAgent, AGUIAgent all have `self.conversation.chat_messages.clear()/extend()`
- **`load_session()` API undefined** — must return messages instead of mutating agent state
- **Stale `_active_runs` entries** — generator cleanup may not execute `finally`; add TTL/weakref fallback
- **Consider combining Phase 2+3** — Phase 2 alone has zero user-visible benefit; Phase 3 is where resource savings appear
- **SessionState consolidation** — overlaps with existing `Session`, `SessionData` models and ServerState's 7+ per-session dicts. Decision: consolidate into single `SessionState` dataclass.

### User Decisions (2026-04-20)

- **Model override**: `AgentRunContext` storage (NOT save/restore)
- **SessionState**: Consolidate — merge ServerState's 7+ per-session dicts into `SessionState`
- **conversation migration**: Hard cut — no dual-path deprecation
- **Route**: Only Phase 1 will be implemented now. Phase 2/3 are deferred to future demand.

## References

- [RFC-0021: Agent Concurrent Execution Safety](../implemented/RFC-0021-agent-concurrent-execution-safety.md) — Per-run isolation via `AgentRunContext`
- [RFC-0026: Per-Session Agent Instances](../implemented/RFC-0026-per-session-agent-isolation.md) — Phase 1: Remove `agent_lock`
- [RFC-0025: Shared Agent Architecture](./RFC-0025-shared-agent-architecture.md) — Phase 3: Single agent, per-session state
- pydantic-ai `Agent.run()` — `message_history` parameter pattern
- pydantic-ai `GraphAgentState` — Internal per-run state pattern

### Key Source Files

- `packages/wolfharness/src/wolfharness/agents/base_agent.py` — `BaseAgent`, session-scoped instance variables
- `packages/wolfharness/src/wolfharness/agents/agent.py` — `Agent`, `_run_stream_once()`, `self.conversation` references
- `packages/wolfharness/src/wolfharness/running/run_context.py` — `AgentRunContext`, per-run isolation
