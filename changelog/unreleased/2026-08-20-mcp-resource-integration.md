# Unified MCP Resource integration

AgentPool now exposes exactly three model-facing MCP Resource tools:
`list_mcp_resources`, `list_mcp_resource_templates`, and `read_mcp_resource`.
Listings preserve MCP metadata and use opaque host cursors; reads require an
explicit server and URI, with structured errors, text truncation, and bounded
binary attachment handling.

Resource-capable MCP providers are negotiated and registered independently of
the `resources.enabled` model-tool gate. Host catalog enumeration and
`ResourceSource` injection use the configured server display name, preserving
same-URI isolation across providers.
