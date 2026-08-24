# Post-merge review fixes for PR #384 (tool-schema-overlap capability)

Fixes five issues identified during code review of the merged tool-schema-overlap
capability PR:

- **P0 — Resource leak**: `BaseAgent.__aexit__` now calls `__aexit__` on
  `_LifecycleCapable` instances in `_external_capabilities`, closing direct
  `MCPClient` connections created by the `_ensure_client` fallback that were
  previously never cleaned up.
- **P1 — Internal URL**: Replaced the hardcoded internal Sany URL in
  `examples/kb_diag_agent.yaml` with `https://mcp.example.com/…`.
- **P1 — Unhandled exception**: `_get_mcp_server_info` now catches and logs
  errors from `cap.list_tools()` on config-defined MCP capabilities instead
  of crashing the entire status endpoint when a client disconnects.
- **P2 — Enum type limitation**: `ParamOverride.enum` relaxed from
  `list[str]` to `list[Any]` so integer/number enums are configurable.
- **P3 — Code duplication**: Added "intentional duplication" comments to
  `_AliasLoader` in all 10 `agentpool*` shim modules.
