---
name: sync-acp-spec
description: Sync the ACP (Agent Client Protocol) schema implementation with the official reference repo by comparing Rust source types against our Python Pydantic models.
---

# Sync ACP Spec

Keep `src/acp/schema/` aligned with the official Agent Client Protocol reference implementation.

## Steps

1. **Clone the reference repo** into a temporary directory:
   ```bash
   tmp=$(mktemp -d)
   git clone --depth 1 https://github.com/agentclientprotocol/agent-client-protocol "$tmp/acp"
   ```

2. **Identify new commits** since last sync:
   ```bash
   git -C "$tmp/acp" log --oneline <SPEC_SYNCED_COMMIT>..HEAD
   ```
   The synced commit hash is stored as `SPEC_SYNCED_COMMIT` in `src/acp/__init__.py`.

3. **Read the upstream Rust source** (the authoritative types):
   - `$tmp/acp/src/agent.rs` — agent-side types: capabilities, auth methods, session state, config options, slash commands, content blocks, session updates, MCP servers
   - `$tmp/acp/src/client.rs` — client-side types: client capabilities, requests, responses
   - `$tmp/acp/src/error.rs` — error codes and `Error` struct
   - `$tmp/acp/src/content.rs` — content block types
   - `$tmp/acp/src/tool_call.rs` — tool call types
   - `$tmp/acp/src/plan.rs` — plan entry types
   - `$tmp/acp/src/rpc.rs` — JSON-RPC message types, method strings
   - `$tmp/acp/src/ext.rs` — extension notifications
   - `$tmp/acp/src/version.rs` — protocol version constant
   - `$tmp/acp/schema/schema.json` and `schema.unstable.json` — JSON Schema (useful for cross-checking)

4. **Compare with our Python implementation** (mapping table):

   | Upstream Rust file | Our Python module |
   |---|---|
   | `agent.rs` (capabilities) | `schema/capabilities.py` |
   | `agent.rs` (auth, common types) | `schema/common.py` |
   | `agent.rs` (session updates) | `schema/session_updates.py` |
   | `agent.rs` (session state) | `schema/session_state.py` |
   | `agent.rs` (slash commands) | `schema/slash_commands.py` |
   | `agent.rs` (MCP servers) | `schema/mcp.py` |
   | `agent.rs` (responses) | `schema/agent_responses.py` |
   | `agent.rs` (requests to client) | `schema/agent_requests.py` |
   | `client.rs` (requests) | `schema/client_requests.py` |
   | `client.rs` (responses) | `schema/client_responses.py` |
   | `client.rs` (capabilities) | `schema/capabilities.py` |
   | `content.rs` | `schema/content_blocks.py` |
   | `tool_call.rs` | `schema/tool_call.py` |
   | `plan.rs` | `schema/agent_plan.py` |
   | `rpc.rs` | `schema/messages.py` |
   | `ext.rs` | `schema/notifications.py` |
   | `error.rs` | `schema/common.py` (`Error`), `exceptions.py` (`RequestError`) |
   | `version.rs` | `schema/__init__.py` (`PROTOCOL_VERSION`) |

5. **Apply updates** to our code:
   - Add new Pydantic model fields matching new Rust struct fields
   - Map `#[cfg(feature = "unstable_*")]` fields to `field: Type | None = None`
   - Add new types / enums / discriminated unions
   - Update `Literal` types for new method strings in `schema/messages.py`
   - Update `__init__.py` exports in both `schema/__init__.py` and `acp/__init__.py`
   - Preserve our `camelCase` alias convention (`alias_generator=to_camel` in base)
   - Do NOT modify protocol logic outside `schema/` (connection handlers, bridge, transports) — only update those if they need to handle new schema types

6. **Wire up new schema types** in the protocol layer (if needed):
   - `agent/connection.py` — add dispatch cases for new agent methods
   - `client/connection.py` — add client-side methods for new requests
   - `bridge/bridge.py` — add bridge handler cases
   - `agent/protocol.py` — add methods to the `Agent` protocol
   - `agent/implementations/testing.py` — add stubs to `TestAgent`
   - `agent/implementations/debug_server/mock_agent.py` — add stubs to `MockAgent`

7. **Wire up in the ACP server** (if needed):
   - `src/wolfharness_server/acp_server/acp_agent.py` — implement new methods, advertise new capabilities
   - `src/wolfharness_server/acp_server/event_converter.py` — emit new session update types
   - `src/wolfharness_server/acp_server/session.py` — store new state

8. **Update the synced commit hash** in `src/acp/__init__.py`:
   ```bash
   git -C "$tmp/acp" rev-parse HEAD
   ```
   Update the `SPEC_SYNCED_COMMIT` constant to the new hash.

9. **Clean up**:
   ```bash
   rm -rf "$tmp"
   ```

## Key conventions

- Rust `#[serde(rename_all = "camelCase")]` → handled by our `alias_generator=to_camel` in `Schema` base
- Rust `#[serde(rename = "_meta")]` → handled by our `convert()` function (`field_meta` → `_meta`)
- Rust `Option<T>` → `T | None = None`
- Rust `#[serde(default)]` booleans → `bool = False` or `bool | None = False`
- Rust `#[serde(tag = "type")]` enums → Pydantic discriminated unions with `Discriminator()`
- Rust `#[serde(untagged)]` variants → `Field(exclude=True)` on the `type` discriminator field
- Rust `Vec<T>` → `Sequence[T]` (for inputs) or `list[T]`
- Rust `HashMap<K, V>` → `dict[K, V]`
- Unstable features (`#[cfg(feature = "unstable_*")]`) → always included, marked with `**UNSTABLE**` docstrings

## What NOT to change

- `connection.py` (base JSON-RPC connection) — only if the wire protocol itself changes
- `transports.py` — transport layer is our own
- `stdio.py` — our process management layer
- `client/` implementations (`DefaultACPClient`, `HeadlessACPClient`, `NoOpClient`) — wolfharness-specific
- `registry/` — our agent registry, not part of ACP spec
- `task/` — our task abstraction
- `tool_call_reporter.py`, `tool_call_state.py` — our extensions
- `filesystem.py` — our filesystem handler implementation
