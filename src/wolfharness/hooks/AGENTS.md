# hooks — Event Interception System

## Where to Look

| Task | File |
|---|---|
| Hook types (callable, command, prompt) | `base.py`, `callable.py`, `command.py`, `prompt.py` |
| AgentHooks container | `agent_hooks.py` |

## Conventions

- **HookAwareTurn is the single choke point**: `HookAwareTurn` mixin is inherited by both `NativeTurn` and `ACPTurn`. All hook firing goes through `Turn.execute()`.
- **Four firing points**: `pre_turn` (blocking), `post_turn` (finally block), `pre_tool_use` (blocking for native, advisory for ACP), `post_tool_use` (can modify output).
- **Priority: deny > ask > allow**: Hooks run in parallel, results combined with this priority.
- **Double-fire guard**: `hooks_fired` set on `AgentRunContext` prevents same hook firing twice per turn. Old `BaseAgent._run_stream_once()` path and new `Turn.execute()` path both check this.
- **`pre_run`/`post_run` removed in v0.5.0**: Use `pre_turn`/`post_turn` only.
- **`as_capability()` removed**: Hooks fire automatically via `HookAwareTurn`. No manual wiring needed.

## Anti-Patterns

- **Calling `agent_hooks.as_capability()`**: Removed in v0.5.0. Hooks fire automatically.
- **Using `pre_run`/`post_run` in YAML**: Removed. Use `pre_turn`/`post_turn`.
- **Bare `async for` bypassing Turn.execute()**: Skips hook firing. Always go through the Turn path.
