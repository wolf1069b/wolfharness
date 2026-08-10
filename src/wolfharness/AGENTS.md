# wolfharness (Core Framework)

## Overview

229 files implementing agent runtimes, message routing, tool management, skill discovery, session orchestration, MCP integration, and codebase mapping. All processing units share the `MessageNode` abstraction.

## Where to Look

| Task | File |
|---|---|
| Agent lifecycle (setup, run, cleanup) | `agents/base_agent.py` |
| Native PydanticAI agent internals | `agents/native_agent/` |
| ACP agent (subprocess JSON-RPC) | `agents/acp_agent/` |
| Per-run state container | `agents/context.py` (`AgentRunContext`) |
| Tool result injection into conversation | `agents/prompt_injection.py` |
| Stream event types (20+ variants) | `agents/events/events.py` |
| Stream event emitter + processors | `agents/events/event_emitter.py`, `processors.py` |
| Core MessageNode abstraction | `messaging/messagenode.py` |
| pydantic-graph step wrapper | `messaging/graph_adapter.py` |
| Signal bridge (anyenv to pydantic-graph) | `messaging/signal_adapter.py` |
| Chat message compaction pipeline | `messaging/compaction.py` |
| AgentPool registry | `delegation/pool.py` |
| Parallel team orchestration | `delegation/team.py` |
| Sequential team orchestration | `delegation/teamrun.py` |
| Talk connections between nodes | `talk/talk.py` |
| Connection registry + graph edges | `talk/registry.py`, `talk/graph_edges.py` |
| Tool base classes + signature parsing | `tools/base.py` |
| Concrete tool implementations | `tool_impls/{bash,read,grep,...}/` |
| RunExecutor (native agent loop) | `orchestrator/run_executor.py` |
| EventBus, SessionController | `orchestrator/core.py` |
| RunHandle lifecycle (RunLoop) | `orchestrator/run.py` |
| Lifecycle dimensions (5 Protocols) | `lifecycle/protocols.py` |
| Lifecycle types (RunState, ToolExecutionRecord) | `lifecycle/types.py` |
| TriggerSource implementations | `lifecycle/triggers.py` |
| Journal implementations (Memory, Durable) | `lifecycle/journal.py` |
| SnapshotStore implementations (Memory, Durable) | `lifecycle/snapshot_store.py` |
| CommChannel implementations (Direct, Protocol) | `lifecycle/comm_channel.py` |
| EventTransport (InProcess) | `lifecycle/event_transport.py` |
| Dimension factory from config | `lifecycle/factory.py` |
| LifecycleConfig Pydantic model | `wolfharness_config/lifecycle.py` |
| Capability base + all capabilities | `capabilities/{function_toolset,filtered_toolset,combined_toolset,...}.py` |
| Skill YAML frontmatter model | `skills/skill.py` |
| Skill as pydantic-ai capability (instructions, tools, MCP) | `skills/capability.py` (`SkillCapability`) |
| Skill MCP server connection lifecycle | `skills/skill_mcp_manager.py` (`SkillMcpManager`) |
| Skill Python tool import from config | `skills/skill_tool_manager.py` (`SkillToolManager`) |
| Skill auto-discovery from paths | `skills/registry.py` |
| Skill wrapped as slash commands | `skills/command.py`, `command_registry.py` |
| Skill URI resolver (`skill://`) | `skills/uri_resolver.py` |
| Skill config models (McpServerConfig, ToolConfig) | `wolfharness_config/skills.py` |
| MCP client + tool bridge | `mcp_server/{client,manager,tool_bridge}.py` |
| Hook types (callable, command, prompt) | `hooks/{base,callable,command,prompt}.py` |
| AgentHooks container (deprecated) | `hooks/agent_hooks.py` |
| Repomap (tree-sitter code mapping) | `repomap/core.py` |
| Session store + models | `sessions/{store,models}.py` |

## Conventions

- **Every node extends MessageNode**: Agents, teams, and the pool itself. Always implement `_step` for pydantic-graph compatibility.
- **Two queue systems**: Native agents use PydanticAI's `PendingMessageDrainCapability`. ACP agents use manual queues (`_post_turn_injections`, `_post_turn_prompts`). M2 adds `CommChannel` feedback loop for `ProtocolChannel` sessions.
- **RunExecutor over bare iteration**: Always use `RunExecutor` to drive native agent runs. Bare `async for node in agent_run:` skips `after_node_run` hooks and breaks message draining.
- **ToolManager is deprecated**: New tools go through native `AbstractCapability` instances. `ToolManager` emits deprecation warnings.
- **Deferred imports for circular safety**: `TYPE_CHECKING` blocks + `from __future__ import annotations`. For truly circular paths (`messagenode` ↔ `team`), defer imports inside function bodies.
- **Signals at step boundaries**: `SignalEmittingGraphRun` maps pydantic-graph transitions to `Talk` signals. Do not emit signals manually from inside steps.
- **RunLoop = RunHandle + dimension injection**: RunHandle is NOT a new class. Its `start()` async generator is the RunLoop. Six pluggable dimensions (TriggerSource, Journal, SnapshotStore, CommChannel, EventTransport) are injected via constructor fields with `__post_init__` defaults.
- **CommChannel owns the Journal**: Every `CommChannel` has a `_journal` reference. `publish()` journals (append/upsert) before delivery. `StateUpdate` events are journaled but NOT published to EventBus.
- **ProtocolChannel bridges protocol servers**: `SessionController` creates `ProtocolTrigger` and `ProtocolChannel` for protocol-handler sessions. Trigger delivers prompts via a queue; Channel publishes events to EventBus.
- **`lifecycle.EventEnvelope` != `orchestrator.event_bus.EventEnvelope`**: Different types with different responsibilities. Lifecycle envelope is for language-agnostic transport serialization.
- **Skills parse YAML frontmatter**: `Skill` model uses `extra="forbid"` to reject unknown keys. Instructions lazy-load from `SKILL.md`.
- **Skills are capabilities**: `SkillCapability` wraps each `Skill` as an `AbstractCapability` providing instructions (`get_instructions`), tools (`get_toolset`), and tool filtering (`get_wrapper_toolset`). Injected in `get_agentlet()` at position 5 (after MCP capabilities).
- **Skill tools come in two flavors**: Python tools declared via `tools` field (`SkillToolConfig` with `import_path` like `"os:getcwd"`) imported eagerly by `SkillToolManager`. MCP servers declared via `mcp_servers` field (`SkillMcpServerConfig` with `command+args` or `url`) connected lazily per-run by `SkillMcpManager`. Both are prefixed with `{skill_name}__tool__` and `{skill_name}__mcp__` respectively.
- **mcp.json companion file**: A `mcp.json` file in the skill directory (using Claude Desktop format `{"mcpServers": {...}}`) takes precedence over the frontmatter `mcp-servers` field. Environment variables (`${VAR}`) are expanded automatically.
- **allowed_tools enforced via FilteredToolset**: The `parsed_allowed_tools()` method parses the space/comma-separated `allowed-tools` frontmatter string. `SkillCapability.get_wrapper_toolset()` wraps the assembled toolset in a `FilteredToolset` that drops tools not in the allowed list.
- **SkillMcpManager has session-scoped lifecycle**: Connections are per `(session_id, server_name)` pair, lazily established on first tool access, with idle timeout (default 5 minutes) and exponential backoff retry (3 attempts). `on_run_ended()` triggers cleanup.
- **One MCP server per capability**: Each `MCPCapability` wraps exactly one server. Use `CombinedToolsetCapability` to combine them.
- **Span instrumentation is mandatory**: All critical-path methods (RunLoop, Turn, delegation, capabilities, lifecycle) MUST use `@logfire.instrument` or `with logfire.span(...)`. Never `asyncio.create_task()` without an active span — it produces orphan traces. See root AGENTS.md "Telemetry & Span Instrumentation" for rules and naming conventions.

## Anti-Patterns

- **`connect_to()` at runtime**: Deprecated. Define connections in YAML `graph:` or `connections:` sections.
- **Bare `async for` in agent loops**: Use `RunExecutor`. The bare pattern silently drops `after_node_run` capability hooks.
- **Mutable state on Agent objects**: `AgentRunContext` is per-execution isolation. Stashing mutable state on the `Agent` instance leaks between runs.
- **Config model imports from core**: Import config types from `wolfharness_config.*`, not `wolfharness.models`, to avoid circular deps.
- **Direct tool code in `tools/`**: Tool framework goes in `tools/`. Concrete implementations go in `tool_impls/`.
- **Accessing `agent_pool` read-only**: Use `host_context` (immutable `HostContext`) instead. The `agent_pool` property emits `DeprecationWarning` as of M2.
- **Mixing `lifecycle.EventEnvelope` with `orchestrator.event_bus.EventEnvelope`**: These are separate types with different roles. Import from `wolfharness.lifecycle.types` for lifecycle transport envelopes; from `wolfharness.orchestrator.event_bus` for internal EventBus envelopes.
- **Blocking calls in async paths**: MCP connections, tool execution, and hooks all expect async methods. Use `anyio` or `asyncio`.
- **Bare `asyncio.create_task()` without span**: Produces orphan traces. Always ensure a `logfire.span` is active at the call site.

## Notes

- **Talk signals backbone**: Every `Talk` instance carries `message_received`, `forwarded`, and `connection_processed` signals. `SignalEmittingGraphRun` bridges these to pydantic-graph.
- **Compaction is a pipeline**: `CompactionPipeline` runs strategies in sequence. Each strategy receives full history and returns a condensed version.
- **EventBus scoped subscriptions**: `"session"` (exact), `"descendants"` (children), `"subtree"` (full subtree), `"all"` (everything). Default is `"session"`.
- **RunHandle cleanup**: `complete_event` fires after all cleanup. `close_session()` awaits it with timeout, then falls back to `cancel_run()`.
- **Crash recovery via journal.resume()**: Detects in-flight Turns by comparing journal entries against snapshot store turn results. Strategy `"mark_interrupted"` skips re-execution; `"retry"` checks tool execution log for idempotency.
- **Tool execution logging in HookAwareTurn**: `_fire_post_tool_hooks()` calls `_log_tool_execution()` which stores a `ToolExecutionRecord` in the Journal. Independent of hooks config.
- **agent_pool deprecated for host_context**: `MessageNode.agent_pool` emits `DeprecationWarning` (M2). Most call sites migrated in M3 (~60), but 18 references remain (primarily ACP server code). Full removal tracked as follow-up before M4. Use `host_context` (immutable `HostContext`).
- **Codemode is a metacall**: `CodeModeCapability` wraps all tools into a single Python execution tool. One tool to rule them all.
- **Skill commands are protocol-agnostic**: `SkillCommand` wraps skills as slash commands working across ACP, AG-UI, and OpenCode without protocol-specific code.
- **SkillCapability injection order matters**: In `get_agentlet()`, skill capabilities are injected at position 5 (after MCP, deferred bridge, approval bridge, and hook capabilities). Each skill produces one `SkillCapability` instance with its own `SkillMcpManager` and `SkillToolManager` — there is one manager tree shared across all skills from the same agentlet creation call.
- **mcp.json format follows Claude Desktop**: The companion file uses `{"mcpServers": {"name": {"command": "...", "args": [...], ...}}}` JSON format. The `_load_mcp_json()` function handles env var expansion and converts entries to `SkillMcpServerConfig` objects. Only filesystem skills (UPath paths) can have companion files — virtual skills (PurePosixPath) cannot.
- **Tool prefixing prevents name collisions**: Python tools get the prefix `{skill_name}__tool__` and MCP tools get `{skill_name}__mcp__`. This ensures tool names from different skills never collide in the agent's tool namespace.
- **Config model lives in separate package**: `wolfharness_config/` exists solely to prevent import cycles with protocol servers.

## Stream Event Types

`RichAgentStreamEvent` (defined in `agents/events/events.py`) is a PEP 695 `type` union of all event variants flowing through the agent stream. Events are published through the `EventBus` and consumed by protocol server converters (`ACPEventConverter`, `EventProcessor`).

**Event Taxonomy**:

| Event | Kind | Purpose |
|---|---|---|
| `RunStartedEvent` | Lifecycle | Agent run started |
| `PartStartEvent` | Stream | Model response part started |
| `PartDeltaEvent` | Stream | Streamed text delta |
| `PartEndEvent` | Stream | Model response part ended |
| `ToolCallStartEvent` | Tool | Tool invocation started |
| `ToolCallProgressEvent` | Tool | Tool execution progress |
| `ToolCallCompleteEvent` | Tool | Tool invocation completed |
| `StreamCompleteEvent` | Lifecycle | Stream completed (with final message) |
| `RunErrorEvent` | Lifecycle | Run errored |
| `SpawnSessionStart` | Subagent | Subagent session spawned |
| `SpawnSessionComplete` | Subagent | Subagent session completed |
| `SessionResumeEvent` | Lifecycle | Session resumed from checkpoint |
| `CompactionEvent` | Lifecycle | Message compaction applied |
| `PlanUpdateEvent` | Plan | Agent plan updated |
| `CustomEvent[T]` | Generic | Custom payload event |
| `UserMessageInsertedEvent` | **System** | **Inserted user message display for steer/followup** |
| `UserPromptEvent` | User | User prompt forwarded |
| `SystemNotificationEvent` | System | System notification (RFC-0056) |

### UserMessageInsertedEvent

A system-level event that carries inserted user message content through the event stream, enabling all steer/followup entry points to trigger user message display in protocol frontends (ACP and OpenCode).

**Fields**:

| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | Session ID where message was inserted |
| `message_id` | `str` | Unique per insertion, used for dedup with protocol handler emission |
| `content` | `str \| list[Any]` | Message content; `str` for text, `list[Any]` for multi-modal (text + images, structured blocks) |
| `delivery` | `Literal["initial", "steer", "followup"]` | How the message was delivered — initial prompt, mid-turn steer, or between-turn followup |
| `source` | `Literal["protocol", "background_task", "internal"]` | Where the message originated — see `source` field mapping below |
| `timestamp` | `float` | UNIX timestamp of insertion (default: `time.time()`) |

**`source` Field Mapping**:

| Call site | `source` value |
|---|---|
| `_route_message()` from protocol handler | `"protocol"` |
| `steer_from_background_task()` | `"background_task"` |
| `steer()` / `followup()` direct call | `"internal"` |
| `_consume_run()` followup-from-queue | `"internal"` |

Internal paths (`"background_task"`, `"internal"`) are always displayed since they have no prior protocol emission. Protocol paths (`"protocol"`) may be deduplicated against the protocol handler's own ad-hoc emission.

**Dedup Mechanism**: The event carries `message_id`, which is shared with the protocol handler's ad-hoc emission path. Protocol handlers generate the ID first, register it in a per-session dedup set, emit the message to the client, then pass the same ID to `send_message()` -> `_route_message()`. The `ACPEventConverter` and `EventProcessor` check the dedup set and skip if the `message_id` is already displayed. The dedup set lives as `dict[str, set[str]]` on `SessionController` (keyed by `session_id`) and is passed to converters as `displayed_message_ids: set[str]`.

**Publication Points**: The event is published from:
- `SessionController._route_message()` — for all routing paths (initial, steer, followup), if EventBus is available
- `steer_from_background_task()` — for internal steer visibility (sync method, uses `asyncio.create_task()`), if EventBus is available
- `_consume_run()` — for followup-from-queue messages picked up from `prompt_queue`, if EventBus is available
- `RunHandle.steer(emit_user_message=True)` — fire-and-forget via `asyncio.create_task()`, as secondary mechanism
- `RunHandle.followup(emit_user_message=False)` — fire-and-forget via `asyncio.create_task()`, default suppressed

When `EventBus` is `None` (standalone `agent.run()` without a protocol server), publication is silently skipped. Display notifications are only meaningful in protocol server contexts.

### RunHandle steer()/followup() emit_user_message Parameter

`RunHandle.steer()` and `followup()` accept the `emit_user_message` parameter to control whether the event is published:

```python
def steer(self, content: str | list[Any], emit_user_message: bool = True) -> None: ...
def followup(self, content: str | list[Any], emit_user_message: bool = False) -> None: ...
```

- `steer(emit_user_message=True)` (default): Publishes `UserMessageInsertedEvent` with `delivery="steer"`, `source="internal"` via `asyncio.create_task()`. Suppress by passing `False`.
- `followup(emit_user_message=False)` (default): Does NOT publish by default. Pass `True` to enable emission. Suppressed by default to avoid redundant display when `_route_message()` or `_consume_run()` already publishes.

Emission uses `asyncio.get_running_loop().create_task()` (fire-and-forget) because `steer()` / `followup()` are synchronous methods (constrained by RFC-0037). The emission helper wraps in `logfire.span("event.user_message_inserted.emit")` to prevent orphan traces. If no event loop is running, the event is silently skipped (`RuntimeError` caught).

### Relationship with SystemNotificationEvent (RFC-0056 / PR #219)

`UserMessageInsertedEvent` and `SystemNotificationEvent` (proposed in RFC-0056 / PR #219) are complementary event types serving different rendering targets:

| Aspect | UserMessageInsertedEvent | SystemNotificationEvent |
|---|---|---|
| Rendering target | `role="user"` message | `ToolPart(tool="system")` notification |
| Trigger | steer/followup entry points | System lifecycle events |
| Dependency | None | None (independent) |
| Dedup scope | Protocol handler ad-hoc emission | N/A (distinct type) |

When both are implemented, `SystemNotificationEvent` for the same steer message should default to suppressed to avoid redundant display — if `UserMessageInsertedEvent` is emitted for a given `message_id`, the `SystemNotificationEvent` for the same content is redundant. This is a forward-looking decision, not a current constraint.
