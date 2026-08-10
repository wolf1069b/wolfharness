# Module Structure

The codebase is organized into focused packages under `src/`:

## wolfharness/ — Core Agent Framework

- `agents/` - Agent implementations (native, ACP)
- `capabilities/` - Native pydantic-ai capability implementations (MCPCapability, FunctionToolsetCapability, CombinedToolsetCapability, SubagentCapability, CodeModeCapability, FilteredToolsetCapability, AgentContext, DelegationService, ResourceSource, entry-point registry)
- `delegation/` - AgentPool orchestration, Team coordination, message routing
- `lifecycle/` - RunLoop lifecycle dimensions (TriggerSource, Journal, SnapshotStore, CommChannel, EventTransport)
- `messaging/` - Message processing, MessageNode abstraction, compaction
- `tools/` - Tool framework and implementations
- `tool_impls/` - Concrete tool implementations (bash, read, grep, etc.)
- `models/` - Pydantic data models and configuration schemas
- `prompts/` - Prompt management and templating
- `storage/` - Interaction tracking and analytics
- `mcp_server/` - MCP server integration
- `running/` - Agent execution runtime
- `sessions/` - Session management
- `hooks/` - Event hooks system
- `observability/` - Logging and telemetry (Logfire)

## wolfharness_config/ — Configuration Models

Separated for clean imports. Contains YAML schema definitions for agents, teams, tools, MCP servers.

## wolfharness_server/ — Protocol Servers

- `acp_server/` - Agent Communication Protocol server
- `opencode_server/` - OpenCode TUI/Desktop server
- `agui_server/` - AG-UI protocol server
- `openai_api_server/` - OpenAI-compatible API server
- `mcp_server/` - Model Context Protocol server

## wolfharness_toolsets/ — Reusable Toolset Implementations

- `builtin/` - Built-in toolsets (code, debug, subagent, file_edit, workers)
- `mcp_discovery/` - MCP server discovery with semantic search
- Specialized toolsets (composio, search, streaming, etc.)

## wolfharness_storage/ — Storage Providers

- `sql_provider/` - SQLAlchemy-based storage
- `zed_provider/` - Zed IDE storage integration
- `claude_provider/` - Claude storage integration
- `opencode_provider/` - OpenCode storage integration

## Other Packages

- `wolfharness_cli/` - Command-line interface
- `wolfharness_commands/` - Command implementations
- `wolfharness_prompts/` - Prompt templates
- `acp/` - Agent Communication Protocol implementation
  - `client/` - ACP client implementations
  - `agent/` - Agent-side protocol implementation
  - `schema/` - Protocol schemas and types
  - `bridge/` - ACP bridge for connecting agents
  - `transports/` - Transport layer (stdio, websocket)
