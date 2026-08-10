# AgentPool Lifecycle Architecture Analysis

> **Status**: Accepted — this design has been implemented in the AgentPool codebase.
> See `docs/explanation/` for the current architecture documentation.

> Date: 2026-07-08
> Status: Research / Pre-design
> Scope: Cross-framework lifecycle comparison to inform AgentPool lifecycle redesign

## Table of Contents

1. [Current AgentPool Lifecycle](#1-current-wolfharness-lifecycle)
2. [ACP v1 vs v2 Protocol Differences](#2-acp-v1-vs-v2-protocol-differences)
3. [Cross-Framework Comparison](#3-cross-framework-comparison)
4. [Key Design Issues](#4-key-design-issues)
5. [Redesign Principles](#5-redesign-principles)
6. [Proposed Target Architecture](#6-proposed-target-architecture)

---

## 1. Current AgentPool Lifecycle

### 1.1 Seven-Phase Lifecycle

| Phase | Core Class | Responsibility |
|---|---|---|
| **Bootstrap** | `AgentPool.__aenter__()` | Initialize MCPManager, SkillsManager, SessionPool, Storage |
| **Session Creation** | `SessionController.get_or_create_session_agent()` | Create per-session agent instance (3 paths: native top-level / native child / ACP) |
| **Run Initiation** | `SessionController.receive_request()` → `RunHandle` | Create/steer/followup RunHandle, fire-and-forget |
| **Turn Execution** | `NativeTurn.execute()` / `ACPTurn.execute()` | Drive pydantic-ai `agent_run.next(node)` loop / ACP client prompt/stream |
| **Event Distribution** | `EventBus.publish()` → subscriber queues | Pub/sub + replay buffer + coalescing |
| **Protocol Consumption** | `ProtocolEventConsumerMixin` | 4 protocol servers (ACP/OpenCode/AG-UI/OpenAI API) share consumer loop |
| **Cleanup** | `SessionController.close_session()` / `AgentPool.__aexit__()` | Close RunHandle → cascade children → clean MCP → exit agent context |

### 1.2 Core Data Flow

```
Client (protocol handler)
  → SessionPool.receive_request()
  → SessionController.receive_request()
    → RunHandle.start(initial_prompt)  [idle/wake/turn async generator]
      → agent.create_turn() → turn.execute()
        → NativeTurn: pydantic-ai iter/next loop
        → ACPTurn: ACP client.prompt() + stream_events()
      → EventBus.publish(session_id, event)
    → _consume_run() drains to completion
  → Protocol consumer (via ProtocolEventConsumerMixin) delivers to client
```

### 1.3 RunHandle — Central Innovation

`RunHandle` owns an idle/wake/turn loop as an async generator:

```
while not self._closing:
    if no current_prompts:
        self._status = idle
        await self._idle_event.wait()         # BLOCK until wake
        current_prompts = _message_queue
        continue

    self._status = running
    turn = agent.create_turn(prompts, run_ctx, message_history)
    publish RunStartedEvent

    for event in turn.execute():
        publish event via EventBus
        yield event
        if StreamCompleteEvent | RunErrorEvent: break

    await child_done_events (30s timeout)
    collect queued_steer_messages → _message_queue

finally:
    self._status = done
    self.complete_event.set()
```

Messaging into the loop:

| Method | Entry | Behavior |
|---|---|---|
| `steer(message)` | Inject into active turn | ASAP injection via `active_agent_run.enqueue("asap")` |
| `followup(message)` | Queue for next turn | Any status (except closing) |
| `close()` | Set `_closing=True`, wake idle | Any status |

### 1.4 Turn Execution — Two Implementations

**NativeTurn** drives pydantic-ai's `agentlet.iter()` + `agent_run.next(node)`:

```
Phase 0: Fire pre_turn hooks (HookAwareTurn)
Phase 1: agentlet.iter(effective_prompts, deps, message_history)
Phase 2: Loop:
  node = agent_run.next_node
  while node != End:
    if ModelRequestNode | CallToolsNode:
      async with node.stream() as stream:
        for event in stream: yield EventMapper(event)
    node = await agent_run.next(node)
Phase 3: Build ChatMessage from agent_run.new_messages()
Phase 4: Yield StreamCompleteEvent
Finally: Fire post_turn hooks
```

**ACPTurn** wraps an ACP client session/prompt call:

```
Phase 0: Fire pre_turn hooks
Phase 1: acp_client.prompt(session_id, content)
Phase 2: acp_client.stream_events(response) → map via acp_to_native_event
Phase 3: acp_client.get_messages(session_id) → accumulate into ChatMessage
Phase 4: Yield StreamCompleteEvent
Finally: Fire post_turn hooks
```

### 1.5 EventBus Architecture

```
publish(session_id, event)
  → wrap in EventEnvelope(source_session_id, event)
  → append to replay_buffer (deque)
  → for each subscriber queue matching session scope:
    → enqueue with overflow policy (drop_oldest / drop_newest / drop_subscriber)

subscribe(session_id, scope="session" | "descendants" | "subtree" | "all")
  → create asyncio.Queue
  → replay historical events from buffer
  → return queue
```

Coalescing happens subscriber-side via `drain_and_merge()`:
- Batches consecutive same-type delta events
- Merges progress events
- PlanUpdateEvents use last-wins

Immediate events (bypass coalescing): RunStartedEvent, RunErrorEvent, RunFailedEvent, StreamCompleteEvent, SpawnSessionStart, CompactionEvent, SessionResumeEvent, ToolCallStartEvent, ToolCallCompleteEvent, ToolCallDeferredEvent, ElicitationDeferredEvent

### 1.6 MCP Lifecycle

- **3 levels of MCP config**: pool-level, agent-level, session-level (+ skill-level as 4th)
- **2 transport pools**: `GlobalConnectionPool` (pool+agent configs) + `SessionConnectionPool` (session+skill configs)
- **2 caching layers**: `_toolset_cache` (global) + `ctx.toolset_cache` (per-session)
- **Special ACP transport**: ACP MCP servers tunnel over ACP JSON-RPC
- **Known race**: `cleanup_session()` can pop `_session_contexts` between `get()` and access in `as_capability()`

### 1.7 Protocol Consumer Lifecycle

`ProtocolEventConsumerMixin` provides canonical consumer lifecycle:

```
1. _before_consumer_loop(session_id)
2. drain_and_merge(stream) → foreach envelope:
   a. if SpawnSessionStart: _on_spawn_session_start()
   b. _handle_event() → ConsumerShutdown to exit
3. Finally: unsubscribe, _after_consumer_loop
```

| Protocol | Scope | Child Consumers | Notes |
|---|---|---|---|
| ACP | `"session"` | Explicit in `_on_spawn_session_start` | Filters `spawn_mechanism="task"` |
| OpenCode | `"session"` | Explicit | Registers ToolPart, creates EventProcessorContext |
| AG-UI | `"session"` | Minimal | Stateless HTTP |
| OpenAI API | `"session"` | Minimal | Stateless HTTP |

### 1.8 Session Checkpoint & Resume

**Checkpoint flow** (during elicitation deferral):
```
handle_elicitation() in AgentContext:
  Create PendingDeferredCall
  Emit ElicitationDeferredEvent
  checkpoint_manager.checkpoint(message_history, pending_calls)
  Update session store status → "checkpointed"
  Register future in ElicitationFutureRegistry
  await future  ← SUSPENDS agent run (local tools)
  # or: raise CallDeferred ← ENDS agent run (MCP tools)
```

**Resume flow** (`SessionPool.resume_session()`):
```
Path A (in-process elicitation): Resolve futures → Agent run continues
Path B (crash recovery, native): Reconstruct agent → Load checkpoint → run_stream()
Path C (crash recovery, ACP): Reconstruct agent → Reopen subprocess → run()
```

---

## 2. ACP v1 vs v2 Protocol Differences

### 2.1 v2 Design Philosophy

v2 is not a rewrite — it's a targeted evolution. Still JSON-RPC 2.0, same envelope types, same SessionId, same `$/cancel_request`, same capability negotiation. Changes target specific aspects deemed suboptimal before a major release.

Gated behind `unstable_protocol_v2` feature flag with comprehensive `conversion.rs` providing `IntoV1`/`IntoV2` bidirectional conversion.

### 2.2 Key Structural Changes

| Dimension | v1 | v2 | Implication for AgentPool |
|---|---|---|---|
| **State reporting** | Implicit (prompt response = completion) | **Explicit `StateUpdate`**: Running/Idle/RequiresAction | Introduce explicit state transition notifications |
| **Message delivery** | Chunks only (ContentChunk) | **Dual model**: chunks + whole-message replacement (UserMessage/AgentMessage/AgentThought) | EventBus event types need whole-message replacement semantics |
| **Tool calls** | Separate ToolCall(create) + ToolCallUpdate(update) | **Unified ToolCallUpdate (upsert)** + ToolCallContentChunk(streaming) | Simplify tool events to single upsert channel |
| **Client I/O** | fs/writeTextFile, fs/readTextFile, terminal/* | **Removed** — agent self-contained tools | Don't depend on client for I/O |
| **Forward compat** | `#[serde(other)]` discards unknown | `Other(OtherSessionUpdate)` **preserves original value** | Event types should preserve forward compat |
| **Diff structure** | Simple path+old_text+new_text | **Structured DiffChange** (add/delete/modify/move/copy) + DiffPatch | Richer diff format support |
| **Session resume** | No replay parameter | **`replay_from: Option<ReplayFrom>`** | Support replay from specified position |
| **Implementation info** | Optional | **Required `info: Implementation`** | Protocol init must carry implementation info |
| **Auth methods** | `"authenticate"` / `"logout"` | **`"auth/login"` / `"auth/logout"`** | Method name rename |
| **Capabilities field** | `client_capabilities` / `agent_capabilities` | Unified to `capabilities` | Flatten capability field |
| **Session modes** | `session/set_mode` | **Removed** | No more mode switching |
| **Session load** | `session/load` | **Removed** | Use `session/resume` instead |
| **Env vars** | `HashMap<String, String>` | `Vec<EnvVariable>` (typed) | Typed env var structs |
| **Plan shape** | `Plan { entries }` | `PlanUpdate { plan: PlanUpdateContent }` (Items/File/Markdown/Other) | Richer plan representation |

### 2.3 Complete v1 → v2 Breaking Changes

1. Auth method rename: `"authenticate"` → `"auth/login"`, `"logout"` → `"auth/logout"`
2. Initialize `info` required (was optional `client_info`/`agent_info`)
3. Capabilities field flattened: `{client,agent}_capabilities` → `capabilities`
4. Auth method type discriminator required (was optional)
5. Removed client filesystem: `fs/writeTextFile`, `fs/readTextFile`
6. Removed client terminal API: all `terminal/*` methods
7. Session response simplified: removed `modes` from `NewSessionResponse`/`ForkSessionResponse`
8. Session config options: `Option<Vec<...>>` → `Vec<...>` (always sent, may be empty)
9. Removed `session/load`
10. Streaming whole messages: added `UserMessage`/`AgentMessage`/`AgentThought` variants
11. StateUpdate replaces implicit completion: explicit Running/Idle/RequiresAction notifications
12. Unified tool call: `ToolCall`(create) + `ToolCallUpdate`(patch) → only `ToolCallUpdate`(upsert)
13. ContentChunk `message_id`: optional in v1, required in v2
14. Restructured Diff: old `old_text`+`new_text` → structured `DiffChange`+optional `DiffPatch`
15. Env vars typed: `HashMap<String,String>` → `Vec<EnvVariable>`
16. Plan content restructured: old `Plan` → new `PlanUpdateContent` (Items/File/Markdown)

### 2.4 Cross-Version Bridging (conversion.rs)

Located at `agent-client-protocol-schema/src/v2/conversion.rs`:
- `IntoV1` trait — converts v2 types to v1
- `IntoV2` trait — converts v1 types to v2
- `IntoV1Many` trait — handles one-to-many mapping (v2 whole-message → multiple v1 chunks)
- Explicit per-field conversion (no JSON serialization round-trip)
- Returns `ProtocolConversionError` when values can't be represented in target version

Key bridging difficulties:
- `StateUpdate` cannot convert to v1 — v1 has no explicit state notification model
- Whole-message updates expand to separate v1 chunks (no replacement semantics)
- `Other`/`Unknown` enum variants spill as errors in v1 direction
- `ContentChunk.message_id` from v2 optional → v1 required (and vice versa)

---

## 3. Cross-Framework Comparison

### 3.1 Frameworks Analyzed

| Framework | Language | Location | Focus |
|---|---|---|---|
| **AgentPool** | Python | `/packages/wolfharness/` | Unified agent orchestration, YAML config, multi-protocol |
| **pydantic-ai** | Python | `/Users/yuchen.liu/src/pydantic-ai/` | PydanticAI agent framework with graph execution |
| **opencode** | TypeScript (Effect-TS) | `/Users/yuchen.liu/src/opencode/` | Code agent with durable event sourcing |
| **pi** | TypeScript | `/Users/yuchen.liu/src/pi/` | Minimal event-stream agent loop |
| **hermes-agent** | Python | `/Users/yuchen.liu/src/hermes-agent/` | Feature-rich agent with learning loop |
| **deer-flow** | Python (LangChain) | `/Users/yuchen.liu/src/deer-flow/` | 26-middleware agent pipeline |
| **claw-code** | Rust + Python | `/Users/yuchen.liu/src/claw-code/` | High-performance Rust CLI agent |
| **oh-my-openagent** | TypeScript | `/Users/yuchen.liu/src/oh-my-openagent/` | Plugin ecosystem on OpenCode/Codex |

### 3.2 Run Loop Architecture Comparison

| Framework | Loop Structure | Core Abstraction | Hook Points | Assessment |
|---|---|---|---|---|
| **AgentPool** | RunHandle idle/wake/turn → Turn.execute() | RunHandle + Turn + EventBus | 4 (pre/post turn/tool) | Good intent, fragmented implementation |
| **pydantic-ai** | Graph.iter() → AgentRun.next(node) | Graph + Step + Capability | **20+** (5 stages × 4 hooks) | Most complete middleware chain |
| **opencode** | SessionRunner.run() → runTurn() → runTurnAttempt() | Effect-TS Fiber + RunCoordinator | Middleware-style | Most mature event sourcing + DI |
| **pi** | runLoop() inner/outer dual loop | Pure event stream (11 event types) | beforeToolCall/afterToolCall | **Most elegant and minimal** |
| **hermes** | run_conversation() 7000-line single function | None | Callback functions | **Anti-pattern** — god object |
| **deer-flow** | LangChain AgentExecutor | 26 middlewares | Per-middleware | **Finest-grained composition** |
| **claw-code** | ConversationRuntime (Rust) | Crate modularization | Plugin lifecycle | Best performance |

### 3.3 Session Management Comparison

| Framework | Session Model | Persistence | Resume/Replay | Concurrency |
|---|---|---|---|---|
| **AgentPool** | SessionPool + SessionController + RunHandle | SQL (SQLAlchemy) | Checkpoint/Resume (elicitation deferred) | Global lock (bottleneck) |
| **pydantic-ai** | GraphAgentState (per-call) | message_history list | conversation_id across runs | None (stateless) |
| **opencode** | SessionV2 + SessionStore | SQLite + event sourcing (durable events) | replayAll() with divergence detection | SessionRunCoordinator (keyed serialization + coalescing) |
| **pi** | JSONL file + tree branching | File system | Branch/fork/clone | Single session |
| **hermes** | SessionDB (SQLite + FTS5) | SQLite | Session resume | Single session |
| **deer-flow** | LangGraph ThreadState | LangGraph state | LangGraph checkpoint | FastAPI gateway |
| **claw-code** | Session (Rust) | JSONL | Session resume | Single session |

### 3.4 MCP Lifecycle Comparison

| Framework | MCP Management | Connection Pool | Cleanup | Special |
|---|---|---|---|---|
| **AgentPool** | MCPManager + 3-level config | Global + Session dual pool | cleanup_session() (has race) | ACP tunnel transport |
| **opencode** | MCP service in packages/opencode | Per-server client | Finalizer: kill descendant PIDs | OAuth + PKCE |
| **pydantic-ai** | MCPOutputToolset | Per-agent | Agent context exit | defer_loading tool hiding |
| **claw-code** | mcp_lifecycle_hardened.rs | Rust managed | Plugin lifecycle | MCP tool bridge |
| **deer-flow** | Sandbox + MCP integration | Sandbox-scoped | Sandbox teardown | Deferred tool filter |

### 3.5 Event/Streaming System Comparison

| Framework | Event System | Persistence | Overflow Policy | Replay |
|---|---|---|---|---|
| **AgentPool** | EventBus (pub/sub + coalescing) | Replay buffer (deque) | drop_oldest/drop_newest/drop_subscriber | Buffer replay only |
| **opencode** | EventV2 (PubSub + SQL) | **Durable event store** (EventTable + EventSequenceTable) | N/A (unbounded) | **replayAll() with divergence detection** |
| **pydantic-ai** | AgentStream + HandleResponseEvent | None | N/A | None |
| **pi** | Typed event stream (11 types) | None | N/A | None |
| **hermes** | Stream callbacks | SQLite session | N/A | Session resume |
| **deer-flow** | LangChain callbacks | Langfuse + LangSmith | N/A | LangGraph checkpoint |

### 3.6 Unique Innovations per Framework

| Innovation | Source | Description | Value for AgentPool |
|---|---|---|---|
| **Pure event-stream loop** | pi | Every lifecycle phase is a typed event, listeners act as barriers | Replace `_RecentAgentRunStream.pull()` pattern |
| **Middleware chain** | deer-flow (26) / pydantic-ai (20+ hooks) | Horizontal composition, each middleware owns one concern | Replace 4 hook points with staged middleware |
| **Event sourcing + Projector** | opencode | Events persisted to SQL, projectors run inside DB transaction | EventBus should support durable mode |
| **RunCoordinator** | opencode | Keyed serialization + coalescing wakeup | Replace global lock + RunHandle idle/wake |
| **Capability middleware chain** | pydantic-ai | Onion-skin middleware, topological sort, before/after/wrap/error hooks | Unify hooks + capabilities + injection |
| **Steering + Follow-up dual queue** | pi | Steering interrupts current tool batch, follow-up waits for natural stop | Replace 4 injection mechanisms |
| **`terminate: true` per tool** | pi | Tool controls whether to skip follow-up LLM call | Fine-grained control |
| **Hashline editing** | oh-my-openagent | Every line tagged with content hash, edits validated against hash | Solve stale-line corruption in edit tool |
| **Deferred tool filtering** | deer-flow | Tool schemas hidden, promoted on demand | Control context window |
| **Durable events + replay** | opencode | Event persistence + replay + divergence detection | Replace EventBus replay buffer |
| **Explicit StateUpdate** | ACP v2 | Running/Idle/RequiresAction explicit notifications | Replace implicit completion signal |
| **Unified ToolCallUpdate (upsert)** | ACP v2 | One channel replaces create + update | Simplify tool events |
| **Learning loop** | hermes | Skill creation from experience, periodic memory nudges | Self-improving skills |
| **Cron scheduler** | hermes | Built-in unattended task execution | Long-running task support |
| **Channel gateway** | hermes / oh-my-openagent | Telegram/Discord/Slack gateways, session lifecycle event dispatch | Channel wake-up support |

---

## 4. Key Design Issues

### 4.1 Dual State Machines on RunHandle

RunHandle carries both:
- `status` (legacy: pending/running/completed/failed/checkpointed) — used by legacy `_start_task/complete/fail/checkpoint`
- `_status` (modern: idle/running/done) — used by modern `start()` loop

Both are mutated in different code paths. `cancel_run_for_session` calls `RunHandle.cancel()` which sets `_status`, but legacy `fail()` sets `status`. Potential state inconsistency.

### 4.2 Four Coexisting Injection Mechanisms

| Mechanism | Scope | Status |
|---|---|---|
| PydanticAI `PendingMessageDrainCapability` | Native agents | Active |
| `TurnRunner` manual queues (`_post_turn_injections`/`_post_turn_prompts`) | ACP agents | Active |
| `PromptInjectionManager` (`inject`/`consume`) | Tool result augmentation | Legacy |
| `RunHandle.steer()` / `RunHandle.followup()` | New unified interface | Active but not fully replacing |

`BaseAgent.inject_prompt()` and `BaseAgent.queue_prompt()` each have 5+ conditional fallback paths depending on agent type, pool presence, and session context.

### 4.3 Circular References

```
RunHandle ←→ AgentRunContext (run_ctx._run_handle = self)
RunHandle → SessionState → agent
run_ctx.steer_callback = self._steer_callback_wrapper
run_ctx.current_task = asyncio.current_task()
```

Object lifetimes must be carefully managed. Fragile.

### 4.4 MCP Lifecycle Complexity

- 3 levels of config (pool/agent/session) + skill-level as 4th
- 2 connection pools (Global + Session)
- 2 caching layers (global + per-session)
- ACP tunnel transport special case
- Known race: `cleanup_session()` pops `_session_contexts` between `get()` and `as_capability()`

### 4.5 EventBus Silent Drops

`overflow_policy: drop_oldest | drop_newest | drop_subscriber` — when consumer is too slow, events are silently dropped. For critical lifecycle events (SpawnSessionStart, RunErrorEvent, checkpoint events), this could lead to undetected data loss.

### 4.6 Global Lock Contention

`SessionController` uses a single `_lock` for all session operations (creation + close). With many concurrent sessions (N × MCP connections), this could become a bottleneck.

### 4.7 Fragile ContextVar Lifetime

`_current_input_provider` is a bare `ContextVar` set at the top of `RunHandle.start()`, relying on undocumented asyncio Task Context isolation behavior. Cannot `reset()` due to race conditions when async generator is GC'd in a different Context.

### 4.8 Dual Purpose `_run_stream_once()`

Handles both standalone mode (creates local EventBus, runs producer/consumer pattern) and Pool mode (delegates to SessionPool). Controlled by `_maybe_pool_stream()` with multiple gating conditions. Implicit routing makes control flow hard to trace.

### 4.9 Protocol Servers Duplicate Child Consumer Logic

`ProtocolEventConsumerMixin._on_spawn_session_start()` is a no-op by default. Each protocol server that supports subagents must override it, but implementations vary. No shared implementation for the common case.

### 4.10 Constructor Pollution

`BaseAgent.__init__()` accepts `input_provider` (deprecated but still present) and `hooks` (migrating from `AgentHooks.as_capability()` to `HookAwareTurn`, both paths active with double-fire guards).

### 4.11 Checkpoint/Resume State Inconsistency

`active → checkpointed → resuming → active` transition has `allow_active_run` flag workaround for in-process elicitation resume, circumventing state validation.

---

## 5. Redesign Principles

### Principle 1: Single State Machine

RunHandle should have one state machine: `idle → running → idle | done`. Deprecate legacy `status` field. Checkpointed is a sub-state of idle, not independent.

### Principle 2: Unified Message Injection

Adopt pi's steering + follow-up dual queue model. Deprecate `PromptInjectionManager`, `TurnRunner` manual queues. `RunHandle.steer()` and `RunHandle.followup()` as sole entry points. For native agents, internally bridge to `PendingMessageDrainCapability`.

### Principle 3: Capability Middleware Chain (from pydantic-ai)

Extend current 4 hook points to staged middleware chain. Each stage has 4 hook types: before / wrap / after / on_error. Stages: `run` → `node` → `model_request` → `tool_validate` → `tool_execute` → `output`. Support topological sort and relative position declaration.

### Principle 4: Event Persistence (from opencode)

EventBus should support optional durable mode. Critical lifecycle events (RunStarted/Completed/Failed/SpawnSessionStart) must not be dropped. Support replay + divergence detection.

### Principle 5: Explicit State Notifications (from ACP v2)

Introduce `StateUpdate` event: `Running | Idle(stop_reason) | RequiresAction`. Replace implicit "StreamCompleteEvent = done" convention.

### Principle 6: MCP Lifecycle Simplification

Unify to 2-level config: pool-level + session-level. Single connection pool + single cache layer. Fix `cleanup_session()` race with per-session lock.

### Principle 7: Protocol Layer Decoupling

Turn should have unified interface, NativeTurn and ACPTurn through same abstraction. ProtocolEventConsumerMixin's child consumer logic should share default implementation. Introduce ACP v2's `StateUpdate` as protocol-agnostic state notification.

---

## 6. Proposed Target Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Protocol Layer                     │
│  ACP v2 │ OpenCode │ AG-UI │ OpenAI API │ MCP Server │
│  (ProtocolEventConsumerMixin + StateUpdate)          │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Session Orchestration                   │
│  SessionPool → SessionController → RunHandle         │
│  (Single state machine: idle→running→idle|done)     │
│  (Steer + Followup dual queue — unified)             │
│  (Per-session locks, not global)                     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Turn Execution Layer                    │
│  Turn (unified interface)                            │
│  ├─ NativeTurn (pydantic-ai graph)                   │
│  └─ ACPTurn (ACP client)                             │
│  + Capability Middleware Chain (6 stages × 4 hooks)  │
│  + Durable EventBus (optional persistence + replay)  │
│  + StateUpdate events (Running/Idle/RequiresAction)  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Resource Layer                          │
│  MCPManager (2-level config, single pool+cache)     │
│  SkillsManager                                       │
│  Storage                                             │
└─────────────────────────────────────────────────────┘
```

---

## Appendix: Key Files Examined

### AgentPool
- `src/wolfharness/delegation/pool.py` — AgentPool entry/exit lifecycle
- `src/wolfharness/orchestrator/run.py` — RunHandle idle/wake/turn loop
- `src/wolfharness/orchestrator/turn.py` — Turn ABC + HookAwareTurn mixin
- `src/wolfharness/orchestrator/session_pool.py` — SessionPool
- `src/wolfharness/orchestrator/session_controller.py` — SessionController
- `src/wolfharness/orchestrator/event_bus.py` — EventBus
- `src/wolfharness/orchestrator/event_mapper.py` — EventMapper
- `src/wolfharness/agents/base_agent.py` — BaseAgent
- `src/wolfharness/agents/context.py` — AgentRunContext + AgentContext
- `src/wolfharness/agents/native_agent/turn.py` — NativeTurn
- `src/wolfharness/agents/acp_agent/turn.py` — ACPTurn
- `src/wolfharness/agents/native_agent/agent.py` — Native Agent
- `src/wolfharness/agents/prompt_injection.py` — PromptInjectionManager
- `src/wolfharness_server/mixins.py` — ProtocolEventConsumerMixin
- `src/wolfharness/mcp_server/manager.py` — MCPManager

### ACP Reference
- `agent-client-protocol-schema/src/v1/` — ACP v1 schema
- `agent-client-protocol-schema/src/v2/` — ACP v2 schema
- `agent-client-protocol-schema/src/v2/conversion.rs` — Cross-version bridge

### pydantic-ai
- `pydantic_ai_slim/pydantic_ai/agent/__init__.py` — Agent class
- `pydantic_ai_slim/pydantic_ai/_agent_graph.py` — Graph definition, 4 nodes
- `pydantic_ai_slim/pydantic_ai/run.py` — AgentRun graph iterator
- `pydantic_ai_slim/pydantic_ai/capabilities/abstract.py` — AbstractCapability
- `pydantic_ai_slim/pydantic_ai/capabilities/combined.py` — CombinedCapability
- `pydantic_ai_slim/pydantic_ai/_tool_execution.py` — Tool execution + 3 strategies
- `pydantic_ai_slim/pydantic_ai/tool_manager.py` — ToolManager

### opencode
- `packages/core/src/session.ts` — SessionV2
- `packages/core/src/session/runner/llm.ts` — Core react loop
- `packages/core/src/session/runner/index.ts` — SessionRunner interface
- `packages/core/src/session/run-coordinator.ts` — RunCoordinator
- `packages/core/src/session/execution/local.ts` — Local execution
- `packages/core/src/session/input.ts` — SessionInput
- `packages/core/src/event.ts` — EventV2
- `packages/core/src/tool/registry.ts` — ToolRegistry
- `packages/opencode/src/mcp/index.ts` — MCP service

### pi
- `packages/agent/src/agent-loop.ts` — Pure event-stream loop
- `packages/agent/src/agent.ts` — Agent class with steering/follow-up

### hermes-agent
- `agent/conversation_loop.py` — 7000-line run loop
- `agent/agent_init.py` — 1834-line constructor
- `agent/tool_executor.py` — Tool dispatch
- `agent/turn_context.py` — Per-turn prologue
- `agent/turn_finalizer.py` — Post-turn finalization
- `acp_adapter/server.py` — ACP adapter

### deer-flow
- `backend/packages/harness/deerflow/agents/lead_agent/agent.py` — LangGraph agent factory
- `backend/packages/harness/deerflow/agents/middlewares/` — 26 middlewares

### claw-code
- `rust/crates/runtime/src/conversation.rs` — Rust conversation runtime
- `rust/crates/runtime/src/session.rs` — Session persistence

### oh-my-openagent
- `packages/openclaw-core/src/dispatcher.ts` — Event dispatch
- `packages/openclaw-core/src/runtime-dispatch.ts` — Session lifecycle mapping
