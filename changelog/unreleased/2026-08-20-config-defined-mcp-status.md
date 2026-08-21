# MCP status endpoint now reports config-defined MCP capabilities

`GET /mcp` (and the equivalent ACP status path) only reported servers
registered with `MCPManager`. MCP servers defined as YAML capabilities
(`type: mcp`, e.g. `knowledge_diag`) live in the agent's
`_external_capabilities` and were invisible to the status endpoint, so
clients rendered them as unavailable even when they were connected and
fully functional.

`BaseAgent._get_mcp_server_info()` now also scans `_all_capabilities`
for `McpServerCap` instances and synthesizes status entries for any that
are not already reported by an `MCPManager`. Connected capabilities
report their live tool list and server info; capabilities without an
established client report as disconnected. The status check never
triggers a lazy connection, matching the existing
`MCPManager.get_server_status()` contract.
