# `tool_arg_sanitize` capability

Adds `ToolArgSanitizeCapability`, which sanitizes invalid-JSON tool call
arguments in message history before **every** model request — closing the
mid-run gap left by the turn-boundary repairs (`inject_cancelled_tool_results`
in `wolfharness/orchestrator/run.py` and `sanitize_tool_call_args_in_messages`
in `wolfharness/orchestrator/event_mapper.py`).

Some models (e.g. deepseek-v4-flash) occasionally emit tool call arguments that
are not valid JSON. pydantic-ai tolerates this locally (wrapping the raw string
as `INVALID_JSON`) so the run does not crash, but the poisoned `ToolCallPart`
stays in message history; when that history is re-serialized to the provider on
a subsequent model request, the provider rejects it with HTTP 400 ("Assistant
tool call function.arguments must be valid JSON"). This capability prevents
that by sanitizing history before every model request.

Configuration:

```yaml
capabilities:
  - type: tool_arg_sanitize
```

Also extracts the shared invalid-JSON predicate `has_invalid_json_args` into
`wolfharness/utils/pydantic_ai_helpers.py`; the local `_has_invalid_json_args`
in `run.py` now delegates to it as a backward-compatible alias.