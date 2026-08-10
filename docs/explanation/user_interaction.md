# User Interaction Architecture

## Core Abstraction

AgentPool uses **InputProvider** to handle user interactions across different execution contexts (CLI, ACP, OpenCode, tests).

### Three-Layer Architecture

```
┌─────────────────────────────────────────────┐
│ Layer 1: Tools (Protocol-Agnostic)         │
│ - question, tool confirmations              │
│ - Only knows MCP types                      │
│ - Calls ctx.handle_elicitation()            │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ Layer 2: Context (Router)                   │
│ - get_input_provider()                      │
│ - Resolution: context → pool → fallback     │
│ - Pure delegation, no protocol logic        │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ Layer 3: InputProvider (Protocol-Specific)  │
│ ┌──────┐ ┌─────┐ ┌──────────┐ ┌──────┐     │
│ │Stdlib│ │ ACP │ │ OpenCode │ │ Mock │     │
│ └──────┘ └─────┘ └──────────┘ └──────┘     │
└─────────────────────────────────────────────┘
```

**Why separate layers**: Different contexts have fundamentally different I/O mechanisms (blocking stdin vs SSE+HTTP vs protocol RPCs). Unifying them would violate their native patterns.

## Providers

### StdlibInputProvider
**Location**: `wolfharness/ui/stdlib_provider.py`  
**Usage**: CLI, fallback  
**Mechanism**: Blocking `input()` calls  
**Limitations**: No async, no rich UI, no multi-select

### ACPInputProvider  
**Location**: `wolfharness_server/acp_server/input_provider.py`  
**Usage**: ACP clients (Goose, Codex)  
**Mechanism**: Maps elicitation → `request_permission()` **[HACK]**  
**Why hacky**: ACP lacks native elicitation, shoehorns questions into permission system  
**Limitations**: Max 4 options, no multi-select, wrong semantics

### OpenCodeInputProvider
**Location**: `wolfharness_server/opencode_server/input_provider.py`  
**Usage**: OpenCode TUI/Desktop  
**Mechanism**: SSE events + HTTP response endpoints  
**Flow**: Create question → broadcast event → await HTTP reply → resolve future  
**Advantages**: Native questions, multi-select, unlimited options, rich descriptions

### MockInputProvider
**Location**: `wolfharness/ui/mock_provider.py`  
**Usage**: Tests  
**Mechanism**: Pre-programmed responses

## OpenCode Flow (Detailed)

```
Tool: question("Which DB?", options=[...])
  ↓
Context: ctx.handle_elicitation(params)
  ↓
Provider: OpenCodeInputProvider.get_elicitation()
  │
  ├─ Generate question_id: "que_12345"
  ├─ Build OpenCode format with options
  ├─ Create asyncio.Future
  ├─ Store in state.pending_questions[id] = {future, ...}
  ├─ Broadcast SSE: QuestionAskedEvent
  └─ await future  # Blocks until HTTP response
  ↓
OpenCode UI receives SSE → shows question dialog
  ↓
User selects "PostgreSQL"
  ↓
POST /question/que_12345/reply {answers: [["PostgreSQL"]]}
  ↓
Route handler: provider.resolve_question(id, answers)
  ↓
future.set_result(["PostgreSQL"])
  ↓
Provider returns: ElicitResult(action="accept", content="PostgreSQL")
  ↓
Tool gets answer: "PostgreSQL"
```

**Key insight**: SSE broadcasts the question, HTTP receives the response. The future bridges the async gap.

## Provider Resolution

```python
context.input_provider          # 1. Explicit (servers set per-session)
  ↓ (if None)
context.pool._input_provider    # 2. Pool default
  ↓ (if None)
StdlibInputProvider()           # 3. Fallback
```

## Current Issues

### 1. Ownership Ambiguity
**Problem**: Can be set on agent, pool, or context with unclear precedence  
**Fix**: Context should **always** own it, resolve at creation time

### 2. Invisible to Observers
**Problem**: Input requests don't appear in event stream  
**Impact**: Can't observe when agent waits, can't replay conversations  
**Fix**: Emit `InputRequestEvent` and `InputResolvedEvent` while still using provider for response

### 3. ACP Elicitation Hack
**Problem**: Uses permissions for questions (semantic mismatch)  
**Options**: 
- Add elicitation to ACP spec
- Accept limitation and document clearly
- Use ACP resources for complex input

## Recommended Evolution

### Phase 1: Fix Ownership
```python
class NodeContext:
    input_provider: InputProvider  # Always set, never None
    
    @classmethod
    def create(cls, node, pool=None, input_provider=None):
        provider = input_provider or pool?._input_provider or StdlibInputProvider()
        return cls(node=node, input_provider=provider)
```

**Benefit**: Clear ownership, no scattered fallback logic

### Phase 2: Add Observability
```python
class InputProvider:
    event_emitter: EventEmitter | None
    
    async def get_elicitation(self, params):
        # Emit for observability
        if self.event_emitter:
            await self.event_emitter.emit(InputRequestEvent.from_params(params))
        
        # Handle via protocol-specific method
        result = await self._handle_elicitation(params)
        
        # Emit resolution
        if self.event_emitter:
            await self.event_emitter.emit(InputResolvedEvent(result))
        
        return result
```

**Benefit**: Input requests visible in stream, no breaking changes

### Phase 3: Bidirectional Streams (Future)
Support optional stream-based resume for advanced providers while keeping async fallback.

## Design Decision: Why Not Pure Event Stream?

**Considered**: Making all interactions part of the bidirectional event stream  
**Rejected because**:
- Different contexts are too different (blocking vs async vs protocol-specific)
- Adds bidirectional complexity to all clients
- Event stream becomes harder to reason about
- Current providers work well for their contexts

**Hybrid approach**: Emit events for observability, use providers for response handling

## Capability Matrix

| Feature | Stdlib | ACP | OpenCode | Mock |
|---------|--------|-----|----------|------|
| Text input | ✅ | ✅ | Future | ✅ |
| Tool confirm | ✅ | ✅ | ✅ | ✅ |
| Boolean | ✅ | ✅ | ✅ | ✅ |
| Single-select | ✅ | ✅ (≤3) | ✅ | ✅ |
| Multi-select | ❌ | ❌ | ✅ | ✅ |
| Descriptions | ❌ | ✅ | ✅ | ✅ |
| Free JSON | ✅ | ❌ | Future | ✅ |
| Async | ❌ | ✅ | ✅ | ✅ |

## Best Practices

**Tools**: Use MCP types, call `ctx.handle_elicitation()`, never check provider type  
**Servers**: Create provider per-session, inject via context  
**Tests**: Use MockProvider with pre-programmed responses  
**Agents**: Set at pool level unless run-specific override needed

## Durable Elicitation

Durable elicitation enables agent sessions to survive process crashes during user interaction. When a tool calls an MCP elicitation (a question to the user), the session state is checkpointed and the run is deferred. After the user responds, the session resumes from the checkpoint -- either in-process (if the agent run is still alive) or through crash recovery (if the process restarted).

### Two-Level Interception Mechanism

The system uses two levels to intercept elicitation requests, because FastMCP's internal callback wrapper catches exceptions:

**Level 1 -- Sentinel**: When `handle_elicitation()` detects a durable provider (`supports_durable_elicitation=True`), it stores the elicitation parameters in a side-channel dict on the context and returns `ElicitResult(action="decline")`. FastMCP sees a normal decline and does not treat it as a failure.

**Level 2 -- Side-channel**: After `MCPClient.call_tool()` returns from the FastMCP call, it checks the side-channel on `AgentContext._pending_elicitation_deferral`. If set, it clears the side-channel and raises `CallDeferred(metadata={"deferred_kind": "elicitation", ...})`. This exception propagates up through pydantic-ai's tool execution loop, triggering checkpoint and deferral.

**Why two levels**: FastMCP's elicitation callback wrapper catches exceptions. If we raised `CallDeferred` inside the handler, FastMCP would swallow it. The sentinel return satisfies FastMCP, and the side-channel check happens after control returns from FastMCP.

### Sentinel + Side-Channel Pattern

The pattern works in three phases:

1. **Inside the handler**: `handle_elicitation()` copies the elicitation params (message, schema, mode) into `_pending_elicitation_deferral` and returns `action="decline"`. The side-channel is a simple dict on `AgentContext`.

2. **After the call returns**: `MCPClient.call_tool()` checks `agent_ctx._pending_elicitation_deferral`. If non-None, it clears the field and raises `CallDeferred` with the elicitation metadata attached. This exception is caught by pydantic-ai's `HandleDeferredToolCalls` capability.

3. **The bridge capability**: `ElicitationDeferredBridge` (registered as a `HandleDeferredToolCalls` capability) intercepts the deferred call, checkpoints the session via `CheckpointManager`, emits an `ElicitationDeferredEvent` to the event bus, and registers a future in `ElicitationFutureRegistry`. The call remains blocked (unresolved) in pydantic-ai's final result.

This diagram shows the flow:

```
Tool: question("Which DB?")
  |
  v
ctx.handle_elicitation(params)
  |-- provider.supports_durable_elicitation? --NO--> provider.get_elicitation() (sync, unchanged)
  |-- YES
  |
  v
Store params in _pending_elicitation_deferral (side-channel)
Return ElicitResult(action="decline") (sentinel)
  |
  v
MCPClient.call_tool() returns
  |-- Check side-channel
  |
  v
CallDeferred(deferred_kind="elicitation", metadata={...})
  |
  v
ElicitationDeferredBridge:
  1. Checkpoint session via CheckpointManager
  2. Emit ElicitationDeferredEvent
  3. Register asyncio.Future in ElicitationFutureRegistry
  4. Return None (call remains blocked)
```

### Two Resume Paths

**In-process resume**: When the agent run is still alive (process did not crash), the `ElicitationFutureRegistry` holds pending futures. `resume_session()` calls `_try_in_process_elicitation_resume()`, which checks the registry for each known `deferred_handle`. If all futures exist, they are resolved with the user's `ElicitationResumePayload`. The MCP client's `call_tool()` RPC unblocks and completes normally. The agent run never restarts -- it was just paused on `await future`.

**Crash recovery resume**: When the process restarted, the registry is empty. `resume_session()` calls `_resume_native_agent()`, which pre-populates `AgentRunContext.cached_elicitation_responses` (keyed by `tool_call_id`) from the elicitation payloads. The agent is reconstructed from checkpoint and re-executed. During MCP tool re-execution, `handle_elicitation()` finds the cached response and returns it instead of deferring again. The MCP tool receives the user's prior response and completes normally.

```
                    resume_session()
                           |
                  elicitation_payloads? --NO--> crash recovery (existing behavior)
                           |
                          YES
                           |
              _try_in_process_elicitation_resume()
                           |
              Futures exist in registry? --YES--> resolve futures in-place
                           |                        (agent run continues)
                          NO
                           |
              _resume_native_agent()
              (pre-populate cached_elicitation_responses,
               re-execute from checkpoint)
```

### Provider Opt-In

`InputProvider` has a `supports_durable_elicitation` property that defaults to `False`. Providers that support checkpoint-based elicitation override it:

- **ACPInputProvider**: Dynamically returns `self.session.checkpoint_enabled`. When checkpointing is on, the ACP session can survive disconnects and restarts.
- **OpenCodeInputProvider**: Dynamically checks whether the session controller's session has `checkpoint_enabled` set. OpenCode clients that support durable sessions opt in automatically.
- **StdlibInputProvider, MockInputProvider**: Inherit the default `False`. These blocking or test-oriented providers always use the synchronous elicitation path.

When `supports_durable_elicitation` is `False`, `handle_elicitation()` calls `provider.get_elicitation()` directly (synchronous, unchanged behavior). When `True`, it uses the deferral path described above.

### Implicit Behavior Shift for get_elicitation()

On durable providers, `get_elicitation()` is called only during crash recovery re-execution to return cached responses. On non-durable providers, `get_elicitation()` is always called for interactive prompting. This means durable providers do not reach `get_elicitation()` during normal operation -- the elicitation is deferred and resolved through the future registry instead.

### Future MRTR Integration Path

The `CheckpointResolutionStrategy` is the current default: it persists state via `CheckpointManager`. A `ProtocolResolutionStrategy` placeholder exists for future MRTR (Model Request Token Resume) support. Here is the planned upgrade path:

**MRTR uses requestState**: Instead of client-side checkpointing (where AgentPool persists and reloads pydantic-ai message history), MRTR would use a `requestState` mechanism where the server holds state across requests. The client sends a state token with each request, and the server reconstructs context from it. This eliminates the need for explicit checkpoint storage.

**SEP-2663 uses tasks/get + tasks/update**: The SEP-2663 proposal defines a task-based protocol for long-running operations. `tasks/get` retrieves the current state of a deferred task, and `tasks/update` submits the user's response. This replaces the application-level checkpoint + resume flow with a standardized protocol exchange.

**ProtocolResolutionStrategy is the extension point**: The `ElicitationResolutionStrategy` protocol lets implementations swap the resolution mechanism. `CheckpointResolutionStrategy` implements the current checkpoint-based approach. `ProtocolResolutionStrategy` (placeholder) will implement the MRTR approach. `resume_session()` dispatches to the appropriate strategy based on what the session and provider support.

**MCP server idempotency**: Crash recovery re-executes MCP tools. The MCP server must safely handle duplicate tool calls with the same inputs. Tools that produce side effects (sending emails, writing files) should be idempotent or the application should tolerate re-execution.
