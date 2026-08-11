# Expand environment variables in MCP HTTP/SSE header values

MCP HTTP/SSE header values now support `${VAR}` environment variable
expansion before the transport is created. Previously, a header such as
`Authorization: Bearer ${API_TOKEN}` was sent to the MCP server as a literal
string, causing `401 Unauthorized` for authenticated servers and forcing users
to hard-code credentials in YAML.

The expansion is applied in `SSEMCPServerConfig.to_transport()`,
`StreamableHTTPMCPServerConfig.to_transport()`, and the session pool's
`_create_transport()`, mirroring the existing `${VAR}` expansion already
supported for skill `mcp.json` companion files.

Resolves wolf1069b/wolfharness#365.