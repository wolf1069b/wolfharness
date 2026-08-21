# Configurable MCP Tool Prefixes

MCP server and Viking capability configuration can now declare an explicit
`tool_prefix` for model-visible tool names. This gives deployments with multiple
knowledge sources a stable, business-owned namespace such as `fault_kb_search`
or `manuals_read`, instead of relying only on automatic display-name-derived
prefixes.

For MCP capability wrappers, `enabled_tools` is adjusted to match the prefixed
tool names exposed to the model. This keeps allow-lists aligned with the actual
toolset after namespacing.
