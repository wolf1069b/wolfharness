# Hooks & Events System

## Hooks

Hooks (`src/wolfharness/hooks/`) intercept agent turns at 4 points: `pre_turn`, `post_turn`, `pre_tool_use`, `post_tool_use`. Three hook types: `CallableHook` (in-process), `CommandHook` (subprocess), `PromptHook` (LLM evaluation). Hooks run in parallel, results combined with priority: deny > ask > allow.

### HookAwareTurn Architecture

Hooks fire through the `HookAwareTurn` mixin (`src/wolfharness/orchestrator/turn.py`), which is inherited by both `NativeTurn` and `ACPTurn`. This provides a single choke point for hook execution inside `Turn.execute()`.

**Firing flow:**

- `fire_pre_turn_hooks()` runs before the LLM call (turn start). Returns `HookResult | None`. If `decision="deny"`, the turn is blocked.
- `fire_post_turn_hooks(result, duration_ms)` runs in the `finally` block after the response, even on error. Injects `duration_ms` from elapsed wall time.
- `fire_pre_tool_hooks(tool_name, tool_input)` runs before a tool call. For native agents, this is blocking (returns `decision="deny"` raises `ModelRetry`). For ACP agents, this is advisory (logs a warning but cannot block).
- `fire_post_tool_hooks(tool_name, tool_output)` runs after a tool call. Can return `modified_output` to replace tool results. For ACP agents, modifies the event payload.
- All four methods check a `hooks_fired` double-fire guard (a `set[str]` on `AgentRunContext`) to prevent the same hook from firing twice in one turn.

**Difference by agent type:**

| Aspect | Native (NativeTurn) | ACP (ACPTurn) |
|--------|--------------------|----------------|
| `pre_turn` | Blocking via `HookAwareTurn` | Blocking via `HookAwareTurn` |
| `post_turn` | Via `HookAwareTurn` finally block | Via `HookAwareTurn` finally block |
| `pre_tool_use` | Blocking via `_ToolInterceptCapability` (model retry on deny) | Blocking via `ACPClientHandler.request_permission()` + advisory on `ToolCallStart` event |
| `post_tool_use` | Via `_ToolInterceptCapability` | Advisory on `ToolCallComplete` event (modify event payload) |
| Standalone fallback | `BaseAgent._run_stream_once()` guarded by `AGENT_TYPE != "native"` | Still uses `BaseAgent._run_stream_once()` (future work to route through `ACPTurn.execute()`) |

**Double-fire guard:** The old path in `BaseAgent._run_stream_once()` fires hooks first and adds keys to `run_ctx.hooks_fired`. The `Turn.execute()` path (called via `_stream_events()`) checks `hooks_fired` and skips if the key is already present. This ensures the old ACP standalone path and the new Turn path don't fire duplicates.

### Deprecated APIs (v0.5.0 removed)

- `AgentHooks.as_capability()` — removed. Hooks now fire automatically via `HookAwareTurn`.
- `pre_run`/`post_run` config field aliases in `HooksConfig` — removed. Use `pre_turn`/`post_turn` only.
- `run_pre_run_hooks()`/`run_post_run_hooks()` methods — removed. Use `run_pre_turn_hooks()`/`run_post_turn_hooks()`.
- `_wrap_before_run()`, `_wrap_after_run()`, `_wrap_before_tool_execute()`, `_wrap_after_tool_execute()` — removed.

### Migration Guide: Hook Rename and HookAwareTurn

**1. YAML config rename** — In your agent YAML config, rename hook fields:

```yaml
# OLD (v0.4.x)
hooks:
  pre_run:
    - type: callable
      callable: mymodule:my_pre_hook
  post_run:
    - type: callable
      callable: mymodule:my_post_hook

# NEW (v0.5.0+)
hooks:
  pre_turn:
    - type: callable
      callable: mymodule:my_pre_hook
  post_turn:
    - type: callable
      callable: mymodule:my_post_hook
```

The old `pre_run`/`post_run` field names were briefly supported with deprecation warnings in v0.4.x and are **removed in v0.5.0**. Only `pre_turn`/`post_turn` are accepted now.

**2. `AgentHooks.as_capability()` removed** — If you programmatically called `agent_hooks.as_capability()` to inject hooks as a PydanticAI capability, remove that call. Hooks now fire automatically through `HookAwareTurn` at the `Turn.execute()` level. No manual wiring is needed.

```python
# OLD (v0.4.x) — removed
capability = agent_hooks.as_capability(hook_manager)

# NEW (v0.5.0+) — hooks fire automatically via HookAwareTurn
# Just configure hooks in YAML or pass AgentHooks to the agent constructor
```

**3. v0.5.0 breaking changes (summary):**

| Removed API | Replacement |
|---|---|
| `HooksConfig.pre_run` / `post_run` | `pre_turn` / `post_turn` |
| `AgentHooks.run_pre_run_hooks()` | `run_pre_turn_hooks()` |
| `AgentHooks.run_post_run_hooks()` | `run_post_turn_hooks()` |
| `AgentHooks.as_capability()` | Automatic via `HookAwareTurn` |
| `_wrap_before_run()`, `_wrap_after_run()`, `_wrap_before_tool_execute()`, `_wrap_after_tool_execute()` | Removed (no replacement — hooks fire via `HookAwareTurn` internally) |

## Events

**Event Types** (`src/wolfharness/agents/events/events.py`): `RichAgentStreamEvent` union type covers streaming deltas, tool calls (start/progress/complete), run lifecycle (started/error/failed), subagent events, session resume, compaction, plan updates, and custom events.

**EventBus** (`src/wolfharness/orchestrator/core.py`): Cross-turn event streaming for protocol servers. Bounded async queues per session, replay buffers, scoped subscriptions (`"session"`, `"descendants"`, `"subtree"`, `"all"`).

**Signal Architecture** (`anyenv.signals.Signal`): In-process type-safe pub/sub on `MessageNode` (`message_received`, `message_sent`) and `Talk` (`connection_processed`, `message_forwarded`). `SignalEmittingGraphRun` bridges pydantic-graph steps to signals.
