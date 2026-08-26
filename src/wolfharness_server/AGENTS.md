# wolfharness_server — Protocol Servers

## Overview

99 .py files implementing 6 protocol servers (ACP, OpenCode, AG-UI, OpenAI API, MCP, A2A) that expose AgentPool agents through standardized wire protocols, plus shared server infrastructure.

## Where To Look

| Task | Location |
|---|---|
| Add/modify a protocol server | `servers/<protocol>/server.py` (entry point), then `base.py` or `http_server.py` for inheritance |
| Understand event consumer lifecycle | `mixins.py` — `ProtocolEventConsumerMixin` (abstract hooks, per-session consumer tasks, locks) |
| Wire a new protocol to SessionPool | Implement `ProtocolEventConsumerMixin`, call `session_controller.receive_request()`, subscribe to EventBus |
| ACP: session create/resume/fork | `acp_server/acp_agent.py` — `AgentPoolACPAgent` handles all ACP `sessions/*` methods |
| ACP: event → ACP notification | `acp_server/event_converter.py` — stateful converter from `RichAgentStreamEvent` to ACP `SessionUpdate` objects |
| ACP: input/confirmation | `acp_server/input_provider.py` — `ACPInputProvider` bridges `InputProvider` to ACP `RequestPermissionResponse` |
| ACP: MCP-over-ACP tunnel | `acp_server/acp_mcp_manager.py` + `acp_server/acp_mcp_transport.py` — bidirectional MCP tunnelled over ACP JSON-RPC |
| ACP: LLM provider metadata | `acp_server/provider_router.py` — derives providers from manifest, tracks overrides |
| ACP: content conversion | `acp_server/converters.py` — pydantic-ai messages ⇄ ACP content blocks |
| ACP: syntax detection | `acp_server/syntax_detection.py` — maps file extensions/dotfiles to language identifiers |
| ACP: slash commands | `acp_server/commands/skill_commands.py` (`ACPSkillBridge`) + `debug_commands.py` + `docs_commands/` |
| ACP: session lifecycle | `acp_server/session.py` (`ACPSession`) + `session_manager.py` (`ACPSessionManager`) |
| OpenCode: event processing pipeline | `opencode_server/event_processor.py` → `stream_adapter.py` → `event_adapter.py` → `state.broadcast_event()` (direct-wire to SSE queues; no EventBus republish) |
| OpenCode: session integration | `opencode_server/session_pool_integration.py` — `OpenCodeSessionPoolIntegration`, main bridge to SessionPool |
| OpenCode: models (20 files) | `opencode_server/models/` — Pydantic models matching OpenCode API types (`parts.py`, `session.py`, `message.py`, `events.py`, etc.) |
| OpenCode: route handlers (14 files) | `opencode_server/routes/` — `session_routes.py`, `message_routes.py`, `file_routes.py`, `config_routes.py`, `agent_routes.py`, etc. |
| OpenCode: state & deps | `opencode_server/state.py` (`ServerState`) + `dependencies.py` (FastAPI `Depends`) |
| OpenCode: input/confirmation | `opencode_server/input_provider.py` — bridges to OpenCode permission system |
| OpenCode: shell command safety | `opencode_server/command_validation.py` — blocks dangerous patterns (rm -rf /, privilege escalation) |
| OpenCode: provider auth | `opencode_server/provider_auth.py` — composable auth backends (OAuth PKCE, device code, API key) |
| OpenCode: skill bridge | `opencode_server/skill_bridge.py` — `SkillCommand` → OpenCode slash commands |
| AG-UI: adapter & skill bridge | `agui_server/base_agent_adapter.py` (`BaseAgentAGUIAdapter`) + `skill_tools.py` (`AGUISkillBridge`) |
| OpenAI API: chat completions | `openai_api_server/completions/helpers.py` + `models.py` |
| OpenAI API: responses API | `openai_api_server/responses/helpers.py` + `models.py` |
| MCP server: FastMCP-based | `mcp_server/server.py` — exposes pool as MCP tools/prompts/resources |
| MCP server: Zed compatibility | `mcp_server/zed_wrapper.py` — parameter encoding for Zed MCP client |
| A2A: types & server | `a2a_server/a2a_types.py` + `server.py` + `agent_worker.py` + `storage.py` |
| Shared utilities | `shared/constants.py` (model defaults) + `shared/model_utils.py` (provider extraction, model info) |
| Multiple protocols at once | `aggregating_server.py` — starts/stops multiple servers sharing one pool |
| HTTP server base class | `http_server.py` — `HTTPServer(base.py:BaseServer)` with Starlette route management |

## Conventions

- **All servers** that subscribe to EventBus extend `ProtocolEventConsumerMixin` from `mixins.py`. Override `_handle_event()` to convert bus events to protocol-specific messages.
- **ACP vs OpenCode model naming**: ACP uses `snake_case` (matching Rust ACP spec). OpenCode uses `camelCase` aliases (`Field(alias="sessionID")`). Both inherit from protocol-specific base models.
- **Route structure** (OpenCode): one router per route file in `routes/`, registered in `opencode_server/server.py`. Use FastAPI `Depends(get_state)` for server state access.
- **Per-session state**: ACP stores state in `ACPSession` objects (managed by `ACPSessionManager`). OpenCode stores state in `ServerState` and `EventProcessorContext` (per-level, supports serialization).
- **Child consumers**: Stateless HTTP servers (AG-UI, OpenAI API) create child consumers on `SpawnSessionStart`. Stateful servers (ACP, OpenCode) do it in `_on_spawn_session_start()`.
- **Skill bridges**: Each protocol has its own bridge (`ACPSkillBridge`, `OpenCodeSkillBridge`, `AGUISkillBridge`) that converts `SkillCommand` to protocol-specific command/tool format.
- **AggregatingServer** manages lifecycle of multiple servers sharing one `AgentPool`. All servers are started/stopped together via `async with AggregatingServer(pool, servers=[...]):`.

## Notes

- **mixins.py is NOT a full protocol server base**. It only handles EventBus consumer lifecycle. Servers also extend `BaseServer` (or `HTTPServer`) for start/stop/run_context management. Multiple inheritance is the norm.
- **ACP `input_provider.py`** is 692 lines — one of the most complex files in the package. It handles tool confirmation dialogs, enum elicitation, and browser-based auth flows.
- **OpenCode `session_pool_integration.py`** is 1442 lines — the largest file. It's the central bridge between OpenCode HTTP routes and SessionPool orchestration. Touch with caution.
- **OpenCode `.rules`** file in the `opencode_server/` directory documents SDK integration conventions, logging setup, and endpoint status. It's a reference doc, not a linter config.
- **ACP `acp_agent.py`** (1111 lines) implements all ACP protocol methods (`sessions/*`, `providers/*`, `tools/*`, etc.) by delegating to the SessionPool and EventBus. It's the ACP entry point.
- **ACP `session.py`** (905 lines) manages `ACPSession` state including MCP server registration, provider config, mode toggles, and the `CommandStore` for slash commands. It bridges to `wolfharness_commands.base.NodeCommand`.
- **ACP `event_converter.py`** (755 lines) is pure conversion logic (no I/O) and is fully testable without mocks. It tracks tool call state internally.
- **OpenCode `event_processor.py`** (798 lines) translates `RichAgentStreamEvent` → OpenCode SSE events with recursive subagent handling. Each level gets its own `EventProcessorContext`.
- **ACP `provider_router.py`** derives provider metadata from `model_variants` in the manifest, not from runtime model discovery. This is a static snapshot at server start.
- **ACP `acp_mcp_manager.py`** does NOT use the standard MCP client. It tunnels MCP JSON-RPC messages over ACP's own session notification mechanism. The companion `acp_mcp_transport.py` bridges this to FastMCP's `ClientTransport`.
- **ACP `commands/debug_commands.py`** provides slash commands (`/replay`, `/inject`, `/checkpoint`, etc.) for replaying ACP notifications during development.
- **ACP `commands/docs_commands/`** provides documentation fetch commands (`/fetch-repo`, `/get-schema`, `/url-to-markdown`, etc.) available as ACP slash commands.
- **OpenCode `command_validation.py`** runs client-side — it's a safety net, not a security boundary. Dangerous patterns are regex-matched before execution.
- **OpenCode `provider_auth.py`** implements Anthropic OAuth PKCE flow for provider authentication. The token store is file-based.
- **AG-UI and A2A** are simple HTTP servers using `BaseAgentAGUIAdapter` — events convert naturally because `RichAgentStreamEvent` is a superset of pydantic-ai's `AgentStreamEvent`.
- **MCP server** does NOT extend `ProtocolEventConsumerMixin`. It uses FastMCP decorators and exposes agents as tools/prompts/resources — no session-level event streaming.
- **A2A server** does NOT extend `ProtocolEventConsumerMixin` either. It's a request-response HTTP server with no event subscription.
- **OpenAI API server** extends both `BaseServer` and `ProtocolEventConsumerMixin`. It uses the mixin only for child consumer lifecycle (subagent sessions). The main chat completions endpoint does not use EventBus directly.

### UserMessageInsertedEvent — ACP Server

The `UserMessageInsertedEvent` (defined in `wolfharness.agents.events.events`) carries inserted user message content through the EventBus. For ACP, this is handled in two places:

**1. `_meta.delivery` extraction at `acp_agent.py:prompt()`**

When a protocol handler calls `handle_prompt()`, `_meta` is already extracted at `acp_agent.py:prompt()` (~line 698) for trace context. The `delivery` field is extracted from `_meta` and passed through the call chain:

- `"steer"` → priority `"asap"` (mid-turn injection)
- `"followup"` → priority `"when_idle"` (between-turn queue)
- Absent → defaults to `"when_idle"`

The delivery value flows through `handle_prompt()` → `send_message()` → `_route_message()` where it determines the priority and populates the `UserMessageInsertedEvent.delivery` field. `message_id` is generated in `handle_prompt()` and passed through the same chain for dedup correlation.

**2. `ACPEventConverter` handling of `UserMessageInsertedEvent`**

`ACPEventConverter` (in `acp_server/event_converter.py`) is modified to accept `protocol_version: int = 1` from the ACP agent (stored at `acp_agent.py:380`). When the converter hits a `UserMessageInsertedEvent`:

```python
case UserMessageInsertedEvent(message_id=mid, content=content):
    if mid in self._displayed_message_ids:
        return  # dedup — already emitted by protocol handler
    blocks = _content_to_blocks(content)
    if self._protocol_version >= 2:
        yield SessionUpdate(
            session_update=SessionUpdateKind.UserMessage,
            message_id=mid, content=blocks,
        )
    else:
        for block in blocks:
            yield SessionUpdate(
                session_update=SessionUpdateKind.UserMessageChunk,
                message_id=mid, content=block,
            )
        self._displayed_message_ids.add(mid)
```

- **ACP v1** (`protocol_version < 2`): Emits one `UserMessageChunk` with `TextContentBlock` per content block. This works with current Zed clients.
- **ACP v2** (`protocol_version >= 2`): Emits a whole-message `UserMessage` upsert with patch semantics. The `UserMessage` model is defined in `src/acp/schema/session_updates.py` with fields `message_id`, `content` (list of ContentBlocks), and `meta`.
- Multi-modal content (`str | list[Any]`) is converted to appropriate `ContentBlock` instances (`str` → `TextContentBlock`, dict with image → `ImageContentBlock`).

**3. Dedup Mechanism**

Dedup uses a per-session `dict[str, set[str]]` on `SessionController` (keyed by `session_id`), passed to `ACPEventConverter` as `displayed_message_ids: set[str]`. The protocol handler (`_emit_user_message_chunks()`) generates the `message_id` first, registers it in the set, emits to the client, then passes the same ID to `send_message(message_id=mid)`. The EventBus event carries the same ID, and the converter checks the set before emitting. Entries are removed on session close.

**4. ACP UserMessage Schema**

The ACP Python schema defines the `UserMessage` Pydantic model in `session_updates.py`:

```python
class UserMessage(BaseModel):
    session_update: Literal["user_message"] = "user_message"
    message_id: str
    content: list[ContentBlock] | None = None
    meta: dict[str, Any] | None = None
```

This is added to the `SessionUpdate` union for v2 protocol support. The existing `UserMessageChunk` is unchanged and remains the v1 path.
