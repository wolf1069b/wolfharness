# orchestrator — RunLoop, Turn, EventBus

## Where to Look

| Task | File |
|---|---|
| EventBus, SessionController | `core.py` |
| RunHandle lifecycle (RunLoop) | `run.py` |
| HookAwareTurn with tool execution logging | `turn.py` |
| RunExecutor (native agent loop) | `run_executor.py` |
| ProtocolTrigger/ProtocolChannel creation | `session_controller.py` |

## Conventions

- **RunLoop = RunHandle + dimension injection**: RunHandle IS the RunLoop. Its `start()` async generator is the idle/wake loop. Six pluggable dimensions are injected via constructor fields with `__post_init__` defaults.
- **RunExecutor over bare iteration**: Always use `RunExecutor` to drive native agent runs. Bare `async for node in agent_run:` skips `after_node_run` hooks and breaks message draining.
- **Dual event publishing**: `event_bus.publish()` (backward-compat path) and `comm_channel.publish()` (M2 path). When CommChannel is `ProtocolChannel`, `start()` avoids double-publishing. `StateUpdate` events are journaled but NOT published to EventBus.
- **Crash recovery via `journal.resume()`**: Detects in-flight Turns by comparing journal entries against snapshot store. `"mark_interrupted"` skips re-execution; `"retry"` checks tool execution log for idempotency.
- **Tool execution logging**: `HookAwareTurn._fire_post_tool_hooks()` calls `_log_tool_execution()` storing a `ToolExecutionRecord` in the Journal. Independent of hooks config — always fires.
- **steer vs followup**: `steer()` injects into active turn (routes to `PendingMessageDrainCapability`). `followup()` queues for next turn, wakes idle event.

## Anti-Patterns

- **Bare `async for` in agent loops**: Use `RunExecutor`. Silently drops `after_node_run` capability hooks.
- **Calling `event_bus.publish()` directly when CommChannel is ProtocolChannel**: Causes double-publish. Let `comm_channel.publish()` handle it.
- **`asyncio.create_task()` without span**: Produces orphan traces. Always ensure `logfire.span` is active at call sites (`_consume_run`, `event_bus.publish`, `_interrupt`).
- **Mutable state on Agent objects**: `AgentRunContext` is per-execution isolation. Stashing on `Agent` leaks between runs.
