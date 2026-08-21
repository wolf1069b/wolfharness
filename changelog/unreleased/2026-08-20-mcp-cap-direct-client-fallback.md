# Config-defined MCP capabilities now connect without a session pool

`McpServerCap` instances built from YAML `type: mcp` capabilities
(via `EntryPointCapabilityConfig.build()`) receive neither a session
connection pool nor a pre-created client. Previously the first tool
access raised `Cannot connect MCP server '<name>': no session pool
configured`, making every config-defined MCP capability unusable.

`McpServerCap._ensure_client()` now falls back to creating a direct
`MCPClient` when neither a client nor a session pool is configured,
mirroring how `MCPManager.setup_server()` creates clients. Raw dict
configs coming from YAML args are additionally normalized to a proper
`MCPServerConfig` at construction time, so the fallback client never
receives a dict (which crashed with `'dict' object has no attribute
'to_transport'`).
