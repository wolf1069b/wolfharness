# Session Orchestration

AgentPool sessions are managed by `SessionPool` and `SessionController`. `SessionPool` holds all active sessions. `SessionController` routes incoming requests and tracks active runs.

## Unified Request Entry Point

`SessionController.receive_request()` is the single entry point for all incoming prompts:

```python
async def receive_request(session_id, content, priority="when_idle")
```

- If the session is idle, it creates a `RunHandle` and starts execution.
- If the session has an active run, it routes based on priority.
- `"asap"` injects into the active turn immediately.
- `"when_idle"` queues the message for the next turn.

Protocol handlers should subscribe to the `EventBus` before calling `receive_request()`, since the method is fire-and-forget. All events stream through the bus.

When creating a new `RunHandle`, `SessionController` wires up two M2 lifecycle dimensions:

- **`ProtocolTrigger`** — An `asyncio.Queue`-based trigger that bridges protocol handlers to the RunLoop. Protocol handlers call `trigger.deliver(content)` to enqueue prompts. The RunLoop drains them in its idle/wake loop via `trigger.poll()`.

- **`ProtocolChannel`** — A bidirectional `CommChannel` that publishes events to the `EventBus` (so protocol servers can consume them) and maintains a feedback queue for steer/followup messages from `SessionController`. `StateUpdate` events are journaled but NOT published to the EventBus, preserving backward compatibility.

For standalone execution (no `EventBus`), `RunHandle.__post_init__` falls back to `DirectChannel` and `ImmediateTrigger`, which deliver events to an internal `asyncio.Queue` that `start()` drains directly.

## Dual Queue Architecture

AgentPool uses two queue systems, with the M2 lifecycle adding a third channel-based path.

**Native agents** rely on PydanticAI's `PendingMessageDrainCapability`. PydanticAI auto-injects this capability outermost. It handles message queuing at two hook points:

- `before_model_request` drains `"asap"` messages immediately before the model call.
- `after_node_run` drains `"when_idle"` messages after the current node finishes.

Native agents drive execution through `RunExecutor`, which calls `agent_run.next(node)` in a loop. The bare `async for node in agent_run:` pattern does not fire `after_node_run` hooks, so `"when_idle"` messages would never drain. `RunExecutor` avoids this by using explicit `next()` calls.

**M2 RunLoop CommChannel** (protocol-server sessions): When `SessionController` creates a `RunHandle`, it injects a `ProtocolChannel` as the CommChannel dimension. `ProtocolChannel` publishes events to the `EventBus` and maintains a feedback queue. `SessionController` calls `channel.deliver_feedback(feedback)` for steer/followup messages. The RunLoop drains feedback via `channel.recv()` in its idle/wake loop.

**DirectChannel** (standalone execution): When no `EventBus` is available, `RunHandle.__post_init__()` creates a `DirectChannel` that publishes events to an internal `asyncio.Queue`. The `start()` generator drains this queue via `get_nowait()`. Standalone execution also uses `ImmediateTrigger` (single-prompt delivery) and relies on `RunHandle._message_queue` for queued prompts between turns.

## RunHandle Lifecycle (RunLoop)

In M2, `RunHandle` IS the RunLoop. It is modified in-place with dimension injection via six optional fields (all set in `__post_init__`):

```python
@dataclass
class RunHandle:
    run_id: str
    session_id: str
    agent_type: str
    agent: BaseAgent[Any, Any] | None
    event_bus: EventBus | None
    session: SessionState | None
    run_ctx: AgentRunContext

    # Lifecycle dimensions (M2) — defaults set in __post_init__
    _trigger_source: TriggerSource | None = None    # ImmediateTrigger("")
    _journal: Journal | None = None                  # MemoryJournal()
    _snapshot_store: SnapshotStore | None = None     # MemorySnapshotStore()
    _comm_channel: CommChannel | None = None         # DirectChannel(journal)
    _event_transport: EventTransport | None = None   # InProcessTransport()
    _lifecycle_session_id: str = "default"
    _run_state: RunState = RunState.IDLE
    _state_lock: asyncio.Lock                        # guards state transitions
    _recover_strategy: str = "mark_interrupted"
```

**`__post_init__()`** initializes any dimension left as `None` to the default in-memory implementation. The journal is injected into the CommChannel so that the channel can persist events. When `SessionController` creates a `RunHandle` for a protocol server session, it passes `ProtocolTrigger` and `ProtocolChannel` explicitly, bypassing the defaults.

**State Machine**: `RunHandle._run_state` transitions through `RunState.IDLE`, `RUNNING`, and `DONE`. Transitions are guarded by `_state_lock` (an `asyncio.Lock`). The `on_state_change()` observer is called on the CommChannel on every transition.

**Lifecycle Flow** (equivalent to the old states):

1. **IDLE** — `RunHandle` created, `_run_state = IDLE`. The `start()` async generator enters the idle/wake loop.
2. **RUNNING** — `start()` receives a prompt and creates a `Turn`. It sets `_run_state = RUNNING` and yields events from `turn.execute()`.
3. **DONE** — `close()` is called, `_run_state = DONE`, and the generator terminates.

Between turns, the handle goes idle and waits on `_idle_event`. Messages arrive via two paths:
- **`steer()`** — Injects a message into the active turn mid-execution (routes to PydanticAI's `PendingMessageDrainCapability`).
- **`followup()`** — Queues a prompt for the next turn. Wakes the idle event.

**Crash Recovery**: When a durable journal and snapshot store are configured, `start()` calls `journal.resume(snapshot_store)` at the beginning:

```python
resume_result = self._journal.resume(self._snapshot_store)
```

If `resume_result.is_inflight` is `True`:
- `"mark_interrupted"` strategy: Marks the interrupted Turn's turn_id on `_recovered_inflight_turn_id`. The RunLoop continues from idle. The interrupted Turn's partial output is preserved in the journal but not re-executed.
- `"retry"` strategy: Same detection, but the caller can check `run_handle.recovered_tool_executions` to see which tools already completed and skip them during re-execution.

During recovery, events since the last snapshot are replayed through the CommChannel with `_replaying = True` (which skips journaling to avoid duplicating entries).

**Legacy methods** (`complete()`, `fail()`, `checkpoint()`) remain for backward compatibility but emit no dimension-driven behavior.

## Session State Machine Mapping

`SessionData.status` (persisted, survives crashes) and `RunState` (transient, in-memory) are separate state machines with a formal mapping defined in `SessionStateMapper` (`src/wolfharness/sessions/state_mapper.py`).

| Scenario | SessionData.status | RunState | Valid? |
|---|---|---|---|
| Idle, no active turn | `active` | `IDLE` | Yes |
| Idle, no RunHandle | `active` | `None` (no handle) | Yes |
| Turn executing | `active` | `RUNNING` | Yes |
| Closed | `closed` | `DONE` | Yes |
| Closed, no RunHandle | `closed` | `None` (no handle) | Yes |
| Checkpointed (post-checkpoint) | `checkpointed` | `None` (no handle) | Yes (do NOT reconcile) |
| Resuming (RunHandle being created) | `resuming` | `IDLE` | Yes |
| Resuming (crash before RunHandle created) | `resuming` | `None` (no handle) | No -> reconcile to `active` |
| Active + crash left no RunHandle | `active` | `None` (no handle) | Yes |

**Invariant checker** (`SessionStateMapper.check_invariant()`) validates consistency at Turn boundaries (snapshot save points). Carve-outs:
- `checkpointed` + no RunHandle = **valid** — do NOT reconcile (normal post-checkpoint state).
- `resuming` + no RunHandle = **reconcile to `active`** (resume failed, session can accept new prompts).
- Other mismatches: log warning + reconcile `SessionData.status` to match `RunState` (in-memory is authoritative for transient state).

## Event Mapping (Native Agents)

`RunExecutor` maps PydanticAI node-level events to AgentPool EventBus events:

| PydanticAI Node Event | AgentPool EventBus Event |
|---|---|
| `AgentRun` created | `RunStartedEvent` |
| `ModelRequestNode` start | `PartStartEvent` |
| `ModelRequestNode` text chunks | `PartDeltaEvent` |
| `ModelRequestNode` end | `PartEndEvent` |
| `FunctionToolCallEvent` | `ToolCallStartEvent` |
| `FunctionToolResultEvent` | `ToolCallCompleteEvent` |
| `EndNode` | `StreamCompleteEvent` |
| Run cancelled | `StreamCompleteEvent(cancelled=True)` |

The `RunExecutor` runs PydanticAI iteration in a background task and pushes events into an async queue. The consumer drains this queue and yields `RichAgentStreamEvent` tokens. This preserves CancelScope safety: cancelling the consumer does not immediately tear down the PydanticAI run.

**M2 Dual Publishing Paths**: Events in the RunLoop are now published via two paths:

- **`event_bus.publish(session_id, event)`** — Backward-compatible path, used by both the old `RunHandle.start()` code and `RunExecutor`. Protocol server `ProtocolEventConsumerMixin` instances subscribe to the EventBus and convert events for their respective protocols.
- **`comm_channel.publish(event)`** — The M2 CommChannel path. `ProtocolChannel.publish()` journals the event (append or upsert) and then publishes to the EventBus. `DirectChannel.publish()` journals the event and enqueues it to an internal `asyncio.Queue`.

When the CommChannel is a `ProtocolChannel`, `start()` avoids double-publishing by NOT calling `event_bus.publish()` directly (detected via `_channel_publishes_to_event_bus`). `StateUpdate` events are journaled but NOT published to the EventBus, since they are internal lifecycle signals that protocol servers do not need to receive.

## PromptInjectionManager

`PromptInjectionManager` serves two purposes depending on the agent type.

**For all agents**, `inject()` and `consume()` handle tool result augmentation. When a tool finishes, `after_tool_execute` hooks call `consume()` to inject additional context into the conversation. If no tool runs, `flush_pending_to_queue()` moves unconsumed injections into the queued prompts.

**For non-native agents**, `queue()` and `pop_queued()` also handle follow-up prompts after a turn ends. The manual queue system drains these through `_post_turn_injections` and `_post_turn_prompts`.

**For native agents**, the follow-up prompt queue (`queue()` / `pop_queued()`) is replaced by PydanticAI's `PendingMessageDrainCapability`. `inject()` / `consume()` remain in use for tool augmentation.
