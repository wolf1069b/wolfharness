# Key Architectural Patterns

## ProtocolEventConsumerMixin

`ProtocolEventConsumerMixin` (in `src/wolfharness_server/mixins.py`) provides a reusable event consumer lifecycle for protocol servers. It extracts the common pattern of subscribing to the `EventBus`, running an async consumer loop, and cleaning up on shutdown.

**Why it exists**: Before this mixin, OpenCode and ACP each implemented their own event consumer loop independently. The code was duplicated, and ACP's implementation was missing features like `SpawnSessionStart` handling and recursive child subscription. The mixin centralizes the loop mechanics while letting each protocol define its own event conversion.

**Which protocols use it**:
- **ACP** (`acp_server/handler.py`): Adopted in Phase 1. Uses `scope="session"` with explicit child consumers created in `_on_spawn_session_start` for sync subagents (skips background tasks with `spawn_mechanism="task"`).
- **OpenCode** (`opencode_server/session_pool_integration.py`): Adopted in Phase 2. Uses `scope="session"` with all hooks implemented (`_before_consumer_loop`, `_handle_event`, `_on_spawn_session_start`, `_after_consumer_loop`). Handles ToolPart registration and `OpenCodeEventAdapter`.
- **AG-UI** (`agui_server/server.py`): Adopted in Phase 3. Uses `scope="session"` with minimal implementation (stateless HTTP, child consumer started in `_on_spawn_session_start`).
- **OpenAI API** (`openai_api_server/server.py`): Adopted in Phase 3. Uses `scope="session"` with minimal implementation (stateless HTTP, child consumer started in `_on_spawn_session_start`).

**Key hooks**:
- `_before_consumer_loop(session_id)`: Set up per-session context (e.g. create an event converter).
- `_handle_event(session_id, event)`: Convert and deliver the event. May raise `ConsumerShutdown` to stop the loop.
- `_on_spawn_session_start(session_id, event)`: React to subagent spawning. Default is no-op.
- `_after_consumer_loop(session_id)`: Clean up per-session context. Only called if the consumer actually started.

**Thread safety**: `start_event_consumer` is idempotent and serializes concurrent calls for the same session via per-session locks.

## MessageNode Abstraction

All processing units (Agents, Teams) inherit from `MessageNode[TInputType, TOutput]`. This provides:
- Unified interface for message processing via `process()`
- Connection management (forwarding outputs between nodes)
- Hook system for intercepting messages
- Type-safe input/output handling

!!! warning "Deprecation: `agent_pool` property"
    `MessageNode.agent_pool` is deprecated since M2 with `DeprecationWarning`. M3.5 completed the migration of all ~64 remaining call sites across ACP server, OpenCode server, core agents, factory, and talk. The `agent_pool` property and setter remain as a deprecated shim for backward compatibility. `HostContext.pool` is kept as a temporary escape hatch for ~6 skill-related accesses (to be removed in the skill-service-extraction change). Full property removal is tracked as an optional task (T7.x) or will be done in M4.

    **Migration**: Use `MessageNode.host_context` instead, which returns an immutable `HostContext` with the same infrastructure fields. `host_context` is sourced from `AgentPool.get_context()` and provides access to MCP manager, storage, and registry without exposing the full mutable pool object.

    ```python
    # OLD (M1) — deprecated
    pool = node.agent_pool
    mcp = node.agent_pool.mcp

    # NEW (M2+) — recommended
    ctx = node.host_context
    mcp = ctx.mcp  # if ctx is not None
    ```

    The `_agent_pool` backing field and setter remain for internal wiring during pool registration, but all read access should migrate to `host_context`.

```python
# Both agents and teams are MessageNodes
agent: MessageNode[ChatMessage, ChatMessage]
team: MessageNode[ChatMessage, TeamRun]

# Nodes can be connected
agent.add_connection(other_agent)  # Forward messages to other_agent
```

!!! warning "Deprecation: Runtime Dynamic Connections"
    `MessageNode.connect_to()` and `ConnectionManager.create_connection()` are deprecated.
    These methods allow runtime mutation of agent topology, which conflicts with the
    immutable graph model used by pydantic-graph.

    **Migration path**: Define connections in YAML (`graph:` or `connections:` sections)
    or use `GraphBuilder` programmatically instead of calling `connect_to()` at runtime.
    The deprecated methods continue to work but will emit a `DeprecationWarning`.

## AgentPool as Registry

`AgentPool` is a `BaseRegistry[NodeName, MessageNode]` that:
- Manages lifecycle of all agents and teams
- Provides dependency injection (shared_deps)
- Handles connection setup from YAML config
- Coordinates resource cleanup

## Team Patterns

**New graph-based approach (recommended):**
Teams are compiled into pydantic-graph workflows:
- **Sequential**: Chained Steps via edges (`agent1 -> agent2 -> agent3`)
- **Parallel**: Fork + Join (`agent1 & agent2 & agent3`)
- **YAML configuration**: Define workflows in the `graph:` section

**Legacy syntax (still supported):**
- **Sequential (chain)**: `agent1 | agent2 | agent3` - Output flows through pipeline
- **Parallel**: `agent1 & agent2 & agent3` - All process same input concurrently
- **YAML configuration**: Define teams in manifest with mode and members

See the Graph Architecture documentation for full details.

## Tool System

Tools follow PydanticAI's tool pattern with AgentPool extensions:
- Tools are typed functions with Pydantic schemas
- `AbstractCapability` is the primary abstraction for providing tools, instructions, and change notifications. Each capability wraps a pydantic-ai `Toolset` and contributes to the agent's compiled tool list.
- Can access `AgentContext` (injected via `RunContext.deps`) for agent-specific state including `DelegationService` for subagent spawning
- Support `subagent` tool for delegation (routes through `DelegationService`, not directly to `AgentPool`)
- Built-in toolsets provide common functionality (code editing, bash, grep)
- `ResourceSource` is a separate protocol for read-only data access (MCP resources, skill content), orthogonal to `AbstractCapability`

## Protocol Bridging

AgentPool acts as a protocol adapter:
1. Agent defined once in YAML (with type: native or acp)
2. Pool loads and manages agent lifecycle
3. Server exposes agent through chosen protocol (ACP/AG-UI/OpenCode/OpenAI API)
4. Client interacts via standardized protocol
