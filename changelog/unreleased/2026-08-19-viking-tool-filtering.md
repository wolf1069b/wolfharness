# Viking tool filtering via enabled_tools / disabled_tools

`VikingCapabilityConfig` gains two mutually-exclusive tool-filter fields,
mirroring `StdioMCPServerConfig`:

- `enabled_tools` — whitelist: only these `viking_*` tools are exposed.
- `disabled_tools` — blacklist: these tools are excluded from the exposed set.

The filter applies in `build_tools()` after mode-based assembly, so it works
for every mode (`retrieve`/`write`/`graph`/`all`) and propagates to both the
toolset and the OpenCode `/experimental/tool` listing via `get_tools()`.

Motivation: the knowledge-graph semantic search backend
(`viking_search`/`viking_find`) can be slow on large resource trees (40-60s+
per query), exceeding typical client timeouts. Operators can now disable
just those tools while keeping the deterministic ones:

```yaml
capabilities:
  - type: viking
    mode: retrieve
    disabled_tools: ["viking_search", "viking_find"]
```

Specifying both `enabled_tools` and `disabled_tools` raises a validation
error (mutually exclusive), matching the MCP server config pattern.