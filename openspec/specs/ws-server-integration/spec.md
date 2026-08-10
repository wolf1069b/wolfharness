## ADDED Requirements

### Requirement: StreamableHTTPTransport configuration type
The `Transport` type union SHALL include `StreamableHTTPTransport` dataclass and the `"streamable-http"` string literal. `StreamableHTTPTransport` SHALL have `host: str = "localhost"` and `port: int = 8080` fields.

#### Scenario: String literal shortcut
- **WHEN** `"streamable-http"` is passed as the transport value
- **THEN** the `serve()` function SHALL normalize it to `StreamableHTTPTransport()` with default host and port

#### Scenario: Custom host and port
- **WHEN** `StreamableHTTPTransport(host="0.0.0.0", port=9000)` is provided
- **THEN** the server SHALL bind to `0.0.0.0:9000`

### Requirement: Transport dispatch in serve()
The `serve()` function SHALL dispatch to `_serve_streamable_http()` when the transport is `StreamableHTTPTransport` or `"streamable-http"`. The match statement SHALL include a case for `StreamableHTTPTransport` that extracts host and port.

#### Scenario: Dispatch with StreamableHTTPTransport
- **WHEN** `serve()` is called with `StreamableHTTPTransport(host="localhost", port=8080)`
- **THEN** the function SHALL call `_serve_streamable_http()` with the provided host and port

### Requirement: ACPServer transport passthrough
`ACPServer.__init__()` and `ACPServer.from_config()` SHALL accept `StreamableHTTPTransport` and `"streamable-http"` as valid transport values and pass them to `acp.serve()` without modification.

#### Scenario: ACPServer with streamable-http transport
- **WHEN** `ACPServer.from_config(config, transport="streamable-http")` is called
- **THEN** the resulting server SHALL use `StreamableHTTPTransport` when starting the ACP agent

### Requirement: CLI --port and --host flags for serve-acp
The `wolfharness serve-acp` CLI command SHALL accept `--port` and `--host` optional flags. When provided, these flags SHALL create a `StreamableHTTPTransport` with the specified values. When neither flag is provided, the CLI SHALL default to stdio transport.

#### Scenario: CLI with --port flag
- **WHEN** the user runs `wolfharness serve-acp config.yml --port 8080`
- **THEN** the CLI SHALL create `StreamableHTTPTransport(port=8080)` and pass it to the ACP server

#### Scenario: CLI with --port and --host flags
- **WHEN** the user runs `wolfharness serve-acp config.yml --host 0.0.0.0 --port 9000`
- **THEN** the CLI SHALL create `StreamableHTTPTransport(host="0.0.0.0", port=9000)`

#### Scenario: CLI without transport flags
- **WHEN** the user runs `wolfharness serve-acp config.yml`
- **THEN** the CLI SHALL use the default stdio transport (no change from current behavior)

### Requirement: YAML config schema extension
The YAML agent configuration schema SHALL support `transport: streamable-http` with optional `host` and `port` fields in the `pool_server` section.

#### Scenario: YAML with streamable-http transport
- **WHEN** a YAML config contains `pool_server: {transport: streamable-http, host: "0.0.0.0", port: 8080}`
- **THEN** `ACPServer.from_config()` SHALL create `StreamableHTTPTransport(host="0.0.0.0", port=8080)`

#### Scenario: YAML with default values
- **WHEN** a YAML config contains `pool_server: {transport: streamable-http}`
- **THEN** `ACPServer.from_config()` SHALL create `StreamableHTTPTransport()` with default host="localhost" and port=8080
