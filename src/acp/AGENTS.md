# ACP Protocol Package

## Overview

Standalone JSON-RPC 2.0 protocol implementation for the Agent Communication Protocol (69 files). Independent from wolfharness framework -- used by it, not part of it.

## Where To Look

| Task | Location |
|---|---|
| Understanding the protocol & types | `schema/` -- all message, session, capability, tool call, terminal types |
| Defining an ACP agent | `agent/protocol.py` (typing.Protocol interface) + `agent/connection.py` (wire protocol) |
| Connecting as a client | `client/protocol.py` (typing.Protocol interface) + `client/connection.py` (wire protocol) |
| Running an agent (stdio/WS/HTTP) | `transports.py` -- `serve()` dispatches to StdioTransport/WebSocketTransport/ACPWebSocketTransport |
| Spawning subprocess agents | `stdio.py` -- `spawn_stdio_connection()`, `spawn_agent_process()`, `spawn_client_process()` |
| Exposing stdio agent over network | `bridge/` -- `ACPBridge` (HTTP), `ACPWebSocketServer` (WS), cli via `acp-bridge` script |
| Low-level message pipeline | `connection.py` -- Connection class with byte streams, observers, dispatcher |
| Task/request lifecycle | `task/` -- MessageQueue, MessageDispatcher, TaskSupervisor, RequestRunner, NotificationRunner |
| Filesystem access over ACP | `filesystem.py` -- `ACPFileSystem` (fsspec), `ACPPath` (upath), entry points in pyproject.toml |
| Debugging & development | `agent/implementations/debug_server/` -- mock agent, debug_server.py, debug.html |
| Standard client implementations | `client/implementations/` -- DefaultACPClient, HeadlessACPClient, NoOpClient |
| Tool call tracking | `tool_call_state.py` -- sends delta-only updates per spec |
| Agent registry | `registry/` -- fetch, archive, model, prepare for agent discovery/management |
| Error types | `exceptions.py` -- `RequestError` with JSON-RPC error codes (-32700 through -32000) |

## Conventions

- **Protocol interfaces use `typing.Protocol`** (not `abc.ABC`). Agent and Client are structural subtyping protocols in `agent/protocol.py` and `client/protocol.py`.
- **JSON-RPC 2.0 over newline-delimited JSON** across all transports. Every message ends with `\n`.
- **Schema-heavy TYPE_CHECKING pattern**: `schema/` has ~20 `# noqa: TC001` suppressions -- TypeCheckType imports are used under `if TYPE_CHECKING` to avoid circular imports.
- **Byte stream abstraction**: All transports convert to `anyio.abc.ByteReceiveStream` / `ByteSendStream`. See the `_WebSocketReadStream` / `_StarletteWebSocketWriteStream` adapters in `transports.py`.
- **Bridge vs native**: Use `acp.serve()` for agents you control. Use `acp.bridge.ACPBridge` only for external stdio agents (see `bridge/README.md` distinction).
- **Factory pattern for agents**: `serve()` accepts `Agent | Callable[[AgentSideConnection], Agent]` -- the factory form lets agents access their connection for sending notifications.
- **Defensive subprocess shutdown**: `spawn_stdio_transport()` in `transports.py` follows MCP SDK pattern -- close stdin first, wait gracefully, then terminate, then kill.
- **W3C trace context via `_meta`**: ACP spec reserves `_meta.traceparent`, `_meta.tracestate`, `_meta.baggage` for W3C trace context ([RFD](https://agentclientprotocol.com/rfds/meta-propagation)). When acting as ACP client, inject `traceparent` via `TraceContextTextMapPropagator.inject()`. When acting as ACP agent, extract it to create child spans linked to the client's trace. Schema models use `field_meta` (aliased to `_meta` in JSON).

## Notes

- The `connection.py` `Connection` class is the central wire-protocol engine. It handles JSON-RPC framing, message dispatch, stream observers, and bidirectional communication. Both `AgentSideConnection` and `ClientSideConnection` wrap it with protocol-specific method handlers.
- `ToolCallState` sends delta-only updates (per ACP spec, all fields except `toolCallId` are optional in updates). Re-sends nothing unless data changed.
- Entry points in pyproject.toml: `acp-bridge` console script, `acp = ACPFileSystem` (fsspec), `acp = ACPPath` (universal_pathlib).
- The debug server at `agent/implementations/debug_server/` provides a full standalone ACP agent + HTML UI for testing without wolfharness.
