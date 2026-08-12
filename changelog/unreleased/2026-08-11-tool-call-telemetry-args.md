# Record tool call arguments in observability spans

Tool-call spans now carry the tool's arguments, which were previously
missing from telemetry: the only instrumentation on the tool execution
path recorded the `tool_name` but not the parameters passed to the tool.

- `ToolInterceptCapability.wrap_tool_execute()` now opens a
  `tool.call` logfire span with `tool_name`, `tool_call_id`, and the
  validated `args` dict. This is the unified interception layer for all
  tool sources (direct, MCP, ACP), so every tool invocation exports its
  arguments regardless of how the tool was registered.
- `ToolDisplayCapability.wrap_tool_execute()` now includes the same
  attributes on its existing instrumentation span.