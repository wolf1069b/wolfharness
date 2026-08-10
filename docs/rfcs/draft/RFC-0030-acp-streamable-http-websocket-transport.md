---
rfc_id: RFC-0030
title: ACP Streamable HTTP WebSocket Transport
status: REVIEW
author: AgentPool Team
reviewers: []
created: 2026-05-13
last_updated: 2026-05-22
decision_date:
related_rfcs: []
---

# RFC-0030: ACP Streamable HTTP WebSocket Transport

## Overview

ACP agents in AgentPool currently communicate over stdio or a bare WebSocket server. Neither transport aligns with the ACP RFC's Streamable HTTP WebSocket Transport profile, which defines a network-accessible endpoint at `/acp` with connection identification, lifecycle enforcement, and HTTP/SSE extensibility. This RFC proposes adding an `ACPWebSocketTransport` built on Starlette and uvicorn that serves the ACP protocol over WebSocket at `/acp`, returns `Acp-Connection-Id` headers, enforces the `initialize` handshake, and integrates with the existing CLI and YAML configuration pipeline. Phase 1 covers WebSocket only; Streamable HTTP (POST/GET/DELETE with SSE) is deferred to Phase 2.

> **Naming Note**: The ACP RFC calls the full profile "Streamable HTTP WebSocket Transport". Phase 1 implements only the WebSocket subset. The dataclass is named `ACPWebSocketTransport` to reflect this, while the CLI/YAML literal `"streamable-http"` refers to the RFC profile name. Phase 2 will extend the same class to support the full profile.

## Table of Contents

1. Background & Context
2. [Problem Statement](#problem-statement)
3. Goals & Non-Goals
4. [Evaluation Criteria](#evaluation-criteria)
5. [Options Analysis](#options-analysis)
6. [Recommendation](#recommendation)
7. [Technical Design](#technical-design)
8. Security Considerations
9. Implementation Plan
10. [Open Questions](#open-questions)
11. Decision Record
12. [References](#references)

## Background & Context

The AgentPool ACP implementation (`src/acp/`) supports three transport modes today:

- **Stdio**: subprocess communication via stdin/stdout. The default and most widely used mode, suitable for IDEs that spawn AgentPool as a child process.
- **WebSocket** (legacy, non-compliant): a bare server using the `websockets` library, defined in `_serve_websocket()`. It accepts connections, wraps them in `AgentSideConnection`, and processes JSON-RPC messages. This transport is **deprecated** and will be removed in v0.6.0 (target: 2026-Q3). Migration: switch to `--transport streamable-http`.
- **Custom streams**: direct `ByteReceiveStream`/`ByteSendStream` injection.

The existing WebSocket transport has significant gaps when measured against the ACP RFC's Streamable HTTP WebSocket Transport profile (`streamable-http-websocket-transport.mdx`):

| Feature | RFC Requirement | Current Status |
|---------|----------------|----------------|
| `/acp` endpoint path | Required | Not enforced; binds to any host:port |
| `Acp-Connection-Id` header | Required on upgrade | Not returned |
| `initialize` lifecycle guard | Required | Not enforced |
| ASGI compatibility | Needed for HTTP routes | Uses raw `websockets` library |
| Extensible to SSE routes | Phase 2 requirement | No HTTP routing possible |

The ACP RFC defines a unified `/acp` endpoint supporting both WebSocket upgrade and Streamable HTTP (POST/GET/DELETE with SSE). Per the RFC, clients MUST support both transports, but servers MAY support only WebSocket. This justifies a phased approach: Phase 1 implements the WebSocket profile, Phase 2 adds Streamable HTTP.

Key existing components that this work builds on:

- `AgentSideConnection` takes `ByteSendStream` + `ByteReceiveStream` and handles the full JSON-RPC line protocol.
- `_WebSocketReadStream` / `_WebSocketWriteStream` adapters already bridge WebSocket frames to anyio byte streams.
- `ACPBridge` is an independent code path (stdio subprocess + HTTP proxy) that requires no changes.
- `uvicorn` is already a project dependency.
- `websockets` is already a project dependency.
- `starlette` would be a new dependency.

## Problem Statement

The current WebSocket transport implementation cannot serve as a compliant ACP server for remote clients. It lacks connection identification, lifecycle enforcement, and the ability to extend to HTTP routes for Streamable HTTP. This blocks several use cases:

1. **IDE integration over the network**: Tools like Zed and VS Code need to connect to an ACP agent running on a remote host or in a container, not just as a subprocess.
2. **Multi-agent orchestration across processes**: Agents running in separate processes or machines need a network transport to communicate.
3. **Protocol compliance**: The ACP RFC's Streamable HTTP WebSocket Transport profile is the standard. The current implementation deviates in multiple ways, making it incompatible with clients that follow the spec.

Without this change, AgentPool can only serve ACP agents via stdio or a non-compliant WebSocket server, limiting its integration surface to in-process or local-subprocess scenarios.

## Goals & Non-Goals

### Goals

- Implement a WebSocket server transport compliant with the ACP RFC's WebSocket profile.
- Return `Acp-Connection-Id` header (UUID v4) during WebSocket upgrade at `/acp`.
- Enforce `initialize` lifecycle: client must send `initialize` before other requests.
- Reuse existing `AgentSideConnection` for ACP protocol handling over WebSocket.
- Integrate with uvicorn/Starlette for production-grade ASGI serving.
- Extend `Transport` type union, `ACPServer`, CLI, and YAML config to support the new transport.
- Document HTTP/1.1 deviation from RFC's HTTP/2 requirement and provide migration path.
- Deprecate the legacy non-compliant `WebSocketTransport` with a migration path.

### Non-Goals

- Streamable HTTP transport (POST/GET/DELETE with SSE). This is deferred to Phase 2.
- Multi-session support over a single WebSocket connection. Phase 1 is one connection equals one session.
- HTTP/2 support. Migration to Hypercorn is planned later; zero app-code change expected.
- Changes to `ACPBridge` or stdio transport.
- Client-side implementation. This RFC covers server-side transport only.
- Authentication or authorization at the transport layer. Phase 1 assumes trusted network.

### Success Criteria

- [ ] ACP server can be started with `wolfharness serve-acp config.yml --transport streamable-http --port 8080` and accepts WebSocket connections at `/acp`
- [ ] WebSocket upgrade response includes `Acp-Connection-Id` header with UUID v4
- [ ] Pre-initialize requests receive JSON-RPC error code `-32002`
- [ ] Post-initialize requests process normally through `AgentSideConnection`
- [ ] Connection cleanup calls `AgentSideConnection.close()` on graceful and abnormal disconnect
- [ ] YAML config `transport: streamable-http` resolves to `ACPWebSocketTransport` instance
- [ ] Legacy `--transport websocket` outputs deprecation warning
- [ ] All integration tests pass

## Evaluation Criteria

The following criteria will be used to objectively evaluate each option:

| Criterion | Weight | Description | Minimum Threshold |
|-----------|--------|-------------|-------------------|
| RFC Compliance | High | Alignment with the ACP RFC's Streamable HTTP WebSocket Transport profile: `/acp` endpoint, `Acp-Connection-Id` header, `initialize` enforcement | Must support all three features |
| Extensibility | High | Ability to add Streamable HTTP transport (POST/GET/DELETE with SSE) in Phase 2 without rewriting the server framework | Must support shared routing, middleware, and connection state |
| Implementation Cost | Medium | Development effort: lines of code, new concepts to learn, testing surface area | Must be achievable in a single sprint |
| Production Readiness | Medium | Middleware support (CORS, logging), monitoring hooks, graceful shutdown, connection lifecycle management | Must support graceful shutdown and at least one middleware |
| Dependency Impact | Low | New dependencies introduced and their cost: bundle size, supply-chain risk, version compatibility | Must not add dependencies with known CVEs |

## Options Analysis

### Option 1: Starlette + uvicorn (WebSocket-only, Phase 1)

**Description**: Build an ASGI application with a Starlette WebSocket endpoint at `/acp`. Serve via uvicorn. Generate `Acp-Connection-Id` header during upgrade. Enforce `initialize` lifecycle via a guard integrated into `AgentSideConnection` (protocol layer, not transport layer). Reuse existing stream adapter patterns adapted for Starlette's WebSocket API.

**Advantages**:

- Standard ASGI app enables adding HTTP routes (SSE, health checks) in Phase 2 without framework changes. Starlette's Router composes WebSocket and HTTP routes naturally.
- Middleware ecosystem is available immediately: CORS, logging, error handling, rate limiting. Any ASGI middleware works.
- uvicorn is already a project dependency, so the HTTP serving layer adds no new runtime requirement.
- Starlette's WebSocket API provides control over upgrade response headers via `websocket.accept(headers=...)`, making `Acp-Connection-Id` return straightforward.
- Production-grade: uvicorn handles process management, signal handling, and graceful shutdown out of the box.

**Disadvantages**:

- Introduces `starlette` as a new dependency. While lightweight and well-maintained, it adds to the dependency tree and supply-chain surface.
- Starlette's WebSocket API differs from the `websockets` library API, requiring new stream adapter implementations rather than reusing the existing ones directly.
- The ASGI abstraction adds a thin layer of indirection compared to the raw `websockets` library. For a single WebSocket endpoint, this indirection is arguably unnecessary in Phase 1.
- HTTP/1.1 only with uvicorn. The RFC may expect HTTP/2. Migration to Hypercorn is possible but introduces another ASGI server later.

**Evaluation Against Criteria**:

| Criterion | Score | Notes |
|-----------|-------|-------|
| RFC Compliance | 4 (Good) | Full control over upgrade headers, path routing, and lifecycle. Can implement all RFC requirements. |
| Extensibility | 5 (Excellent) | ASGI app can add HTTP routes and middleware. Phase 2 Streamable HTTP is a route addition, not a rewrite. |
| Implementation Cost | 3 (Adequate) | New stream adapters needed for Starlette API. Moderate effort for ASGI app scaffolding. |
| Production Readiness | 4 (Good) | Middleware, graceful shutdown, and monitoring all available through ASGI/uvicorn ecosystem. |
| Dependency Impact | 3 (Adequate) | One new dependency (starlette). Lightweight, no Pydantic requirement, widely used. |

**Effort Estimate**: Medium. Approximately 200-300 lines of new code for the ASGI app, stream adapters, initialize guard, and transport config. Integration with existing ACPServer and CLI is mechanical.

**Risk Assessment**:

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Starlette API drift between versions | Low | Low | Pin starlette version in pyproject.toml; Starlette follows semantic versioning. |
| HTTP/1.1 deviation causes client issues | Low | Medium | Document deviation; WebSocket upgrade works over HTTP/1.1; migration path to Hypercorn. |
| Stream adapter bugs in Starlette-WebSocket-to-anyio bridge | Medium | Medium | Write integration tests covering connect, disconnect, and abnormal-close scenarios. |
| Starlette adds transitive dependency weight | Low | Low | Starlette has minimal dependencies (no Pydantic). Audit dependency tree before pinning. |

---

### Option 2: Raw `websockets` library (enhance existing)

**Description**: Extend the current `_serve_websocket()` implementation to add `Acp-Connection-Id` header return and `initialize` enforcement. No ASGI framework. The `websockets` library provides hooks for custom headers during upgrade via its `process_request` callback.

**Advantages**:

- No new dependencies. The `websockets` library is already in the project.
- Minimal code changes to the existing `_serve_websocket()`. Incremental enhancement of working code.
- Familiar codebase. The team already understands the `websockets` library patterns used in the current implementation.
- Lower risk of regression since the existing implementation continues to work; changes are additive.

**Disadvantages**:

- No path to Streamable HTTP. The `websockets` library handles only WebSocket connections. Adding HTTP routes for SSE in Phase 2 would require a separate HTTP server framework, resulting in two parallel server implementations.
- No ASGI middleware. CORS, logging, rate limiting, and other cross-cutting concerns must be implemented manually as custom code within the `websockets` handler.
- The `websockets` library's `process_request` hook for custom headers is less ergonomic than Starlette's direct header access. It requires understanding the library's internal upgrade flow.
- Two server frameworks in the codebase (raw `websockets` for legacy, ASGI for Phase 2) create maintenance burden and conceptual overhead.
- Limited production tooling. The `websockets` library does not provide process management, signal handling, or monitoring hooks comparable to uvicorn.

**Evaluation Against Criteria**:

| Criterion | Score | Notes |
|-----------|-------|-------|
| RFC Compliance | 3 (Adequate) | Can add `Acp-Connection-Id` and initialize guard, but no path routing enforcement. |
| Extensibility | 1 (Inadequate) | Adding HTTP routes requires a separate server. Two frameworks in the codebase. |
| Implementation Cost | 5 (Excellent) | Smallest change delta. Enhance existing code. |
| Production Readiness | 2 (Poor) | No middleware, no process management, limited monitoring. Must build manually. |
| Dependency Impact | 5 (Excellent) | No new dependencies. |

**Effort Estimate**: Low. Approximately 50-80 lines of changes to existing `_serve_websocket()` for header return and initialize guard.

**Risk Assessment**:

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Phase 2 requires full rewrite | High | High | Accept technical debt now, plan rewrite later. Or reject this option. |
| Custom header hook API is fragile | Medium | Medium | Write tests for `process_request` hook; pin `websockets` version. |
| Dual-framework maintenance burden | High | Medium | Document both code paths; plan deprecation of raw `websockets` server. |
| Missing middleware for production use | High | Medium | Build custom middleware as needed, increasing code in transports.py. |

---

### Option 3: FastAPI + uvicorn

**Description**: Use FastAPI's WebSocket support with Pydantic models for protocol message validation. Define request/response schemas for `initialize`, `session/new`, and other ACP methods. Serve via uvicorn.

**Advantages**:

- Automatic OpenAPI documentation generation for any HTTP routes added in Phase 2.
- Pydantic validation on JSON-RPC messages catches malformed requests before they reach `AgentSideConnection`.
- FastAPI is widely known in the Python ecosystem, potentially reducing onboarding time.
- Same ASGI foundation as Option 1, so middleware and routing benefits apply.

**Disadvantages**:

- FastAPI's WebSocket support does not perform Pydantic validation on WebSocket messages. The validation benefit is limited to HTTP routes, not the WebSocket transport that Phase 1 actually needs.
- Adds both `fastapi` and its `pydantic` dependency (if not already present at the required version). This is heavier than Starlette alone.
- FastAPI wraps Starlette, adding an abstraction layer that provides no value for a pure WebSocket endpoint. The overhead shows up in import time, startup time, and complexity.
- `AgentSideConnection` already handles JSON-RPC message parsing and validation. Adding Pydantic validation at the transport layer duplicates this logic and creates two validation points that can drift out of sync.
- FastAPI's opinionated patterns (dependency injection, path operation decorators) add ceremony that a single WebSocket endpoint does not need.

**Evaluation Against Criteria**:

| Criterion | Score | Notes |
|-----------|-------|-------|
| RFC Compliance | 4 (Good) | Same header control and routing as Starlette. |
| Extensibility | 5 (Excellent) | Full ASGI app with HTTP routes and middleware. |
| Implementation Cost | 2 (Poor) | FastAPI boilerplate plus Pydantic message models that duplicate existing validation. |
| Production Readiness | 4 (Good) | ASGI middleware and uvicorn benefits, plus OpenAPI docs. |
| Dependency Impact | 2 (Poor) | Two new dependencies: fastapi plus pydantic version alignment. |

**Effort Estimate**: Medium-High. Approximately 250-350 lines for the FastAPI app, Pydantic message models, and stream adapters. The Pydantic models for JSON-RPC messages add significant upfront work that duplicates existing validation.

**Risk Assessment**:

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Pydantic validation drift from AgentSideConnection | High | Medium | Keep validation in one place (AgentSideConnection); don't duplicate in FastAPI. But this removes the primary benefit of Option 3. |
| FastAPI version coupling with Pydantic v2 | Medium | Medium | Pin both versions; test upgrade paths. |
| Unnecessary abstraction for a WebSocket endpoint | High | Low | Accept added complexity, or simplify to Starlette. |
| Import/startup time regression | Low | Low | Measure impact; FastAPI's import chain is heavier than Starlette's. |

---

### Options Comparison Summary

| Criterion | Weight | Option 1 (Starlette) | Option 2 (Raw websockets) | Option 3 (FastAPI) |
|-----------|--------|----------------------|--------------------------|-------------------|
| RFC Compliance | High | 4 | 3 | 4 |
| Extensibility | High | 5 | 1 | 5 |
| Implementation Cost | Medium | 3 | 5 | 2 |
| Production Readiness | Medium | 4 | 2 | 4 |
| Dependency Impact | Low | 3 | 5 | 2 |
| **Weighted Total** | | **3.9** | **2.8** | **3.6** |

---

## Recommendation

### Recommended Option

**Option 1: Starlette + uvicorn**

### Justification

The deciding factor is Extensibility, weighted High. Phase 2 requires adding Streamable HTTP routes (POST/GET/DELETE with SSE) alongside the WebSocket endpoint. Starlette's ASGI Router composes these naturally: the WebSocket route at `/acp` and HTTP routes at the same path share middleware, connection state, and the server framework. Option 2 (raw `websockets`) cannot serve HTTP routes, meaning Phase 2 would require introducing a second server framework or rewriting the transport entirely. Option 3 (FastAPI) provides the same extensibility but adds Pydantic validation overhead that the WebSocket transport cannot use, since FastAPI does not validate WebSocket message payloads.

On RFC Compliance, Options 1 and 3 score equally (4/5), while Option 2 scores lower (3/5) due to the inability to enforce path-based routing cleanly.

On Implementation Cost, Option 2 scores best (5/5) because it is the smallest change. However, this cost advantage is temporary: the Phase 2 rewrite needed under Option 2 would exceed the total cost of Option 1's Phase 1 plus Phase 2.

On Production Readiness, Option 1 scores well (4/5) due to ASGI middleware support, uvicorn's process management, and graceful shutdown. Option 2 scores poorly (2/5) because these capabilities must be built manually.

On Dependency Impact, Option 2 has no new dependencies. Option 1 adds starlette (lightweight, no Pydantic). Option 3 adds both fastapi and pydantic version constraints. Starlette is the smallest addition among the options that provide ASGI extensibility.

### Accepted Trade-offs

1. **HTTP/1.1 deviation**: Acceptable because WebSocket upgrade works over HTTP/1.1. The deviation is functionally harmless for Phase 1. Migration to Hypercorn (ASGI-compatible HTTP/2 server) requires zero app-code changes; only the server runner call changes.
2. **New starlette dependency**: Acceptable because starlette is lightweight (~30KB, no Pydantic), well-maintained (sibling project of FastAPI), and provides the ASGI foundation needed for Phase 2. The alternative of maintaining two server frameworks (Option 2 + separate HTTP server for Phase 2) carries higher long-term cost.

### Conditions

- HTTP/1.1 deviation must be documented in server startup logs.
- starlette dependency must be audited for transitive dependencies before pinning.
- Initialize enforcement must be configurable (allow opt-out for backward compatibility during migration).

---

## Technical Design

### Architecture Overview

```
                          ACP Streamable HTTP WebSocket Transport
                          ======================================
  Client (IDE, CLI, etc.)
       |
       |  HTTP/1.1 Upgrade: ws://host:port/acp
       |  Response Header: Acp-Connection-Id: <uuid-v4>
       v
  +----------------------------------------------------------+
  |                   Starlette ASGI App                      |
  |                                                           |
  |  WebSocket Route: /acp                                    |
  |  +-----------------------------------------------------+ |
  |  |                                                     | |
  |  |  1. Accept upgrade, generate Acp-Connection-Id     | |
  |  |  2. Create stream adapters                         | |
  |  |     - _StarletteWebSocketReadStream -> ByteRecv    | |
  |  |     - _StarletteWebSocketWriteStream -> ByteSend   | |
  |  |  3. Create AgentSideConnection(agent_factory, ...) | |
  |  |     - Initialize guard lives in Connection layer   | |
  |  |  4. Monitor connection lifecycle                   | |
  |  |     - Graceful close -> AgentSideConnection.close()| |
  |  |     - Abnormal close -> AgentSideConnection.close()| |
  |  +-----------------------------------------------------+ |
  |                                                           |
  |  (Phase 2: HTTP Route: /acp for Streamable HTTP/SSE)     |
  +----------------------------------------------------------+
       |
       |  uvicorn serves the ASGI app
       v
  Network (host:port)
```

### Key Components

#### ACPWebSocketTransport

- Responsibility: Configuration dataclass for the transport
- Fields: `host: str = "localhost"`, `port: int = 8080`
- Type: `@dataclass`
- Note: Phase 1 is WebSocket-only. Phase 2 will extend this class to support Streamable HTTP (POST/GET/DELETE with SSE).

#### _StarletteWebSocketReadStream

- Responsibility: Adapt Starlette WebSocket receive to `ByteReceiveStream`
- Pattern: Wraps `WebSocket.receive_text()`, translates `WebSocketDisconnect` to `anyio.EndOfStream`, appends trailing newline for JSON-RPC line protocol
- Base: `ByteReceiveStream`

#### _StarletteWebSocketWriteStream

- Responsibility: Adapt Starlette WebSocket send to `ByteSendStream`
- Pattern: Strips trailing newline, sends as complete WebSocket text message via `WebSocket.send_text()`
- Base: `ByteSendStream`

#### Initialize Guard (Protocol Layer)

- Responsibility: Enforce `initialize` as the first JSON-RPC request after WebSocket upgrade
- Location: Integrated into `AgentSideConnection` / `Connection` layer, NOT the transport layer
- Rationale: The transport layer should not duplicate JSON-RPC parsing logic. `Connection._receive_loop()` already parses JSON-RPC messages. The guard intercepts at the protocol layer after parsing.
- State: `initialized: bool` per connection
- Behavior:
  - Before initialize: Any request with `method != "initialize"` returns JSON-RPC error `{"jsonrpc":"2.0","error":{"code":-32002,"message":"initialize required"},"id":<request_id>}`
  - After successful initialize: All messages pass through unmodified
  - Batch requests: If the first message is a batch, the entire batch is rejected with `-32002` unless all elements are `initialize` (Phase 1 does not support batching)
- Implementation notes:
  - The guard inspects the parsed JSON-RPC request object (already available in `Connection._process_message()`). No additional stream parsing is needed.
  - To detect a successful `initialize` response, the guard wraps the agent's `initialize()` handler. When the handler returns an `InitializeResponse` (or the response is serialized successfully), the guard flips `initialized = True`. If the handler raises an exception, the connection remains uninitialized and the client may retry.
  - The guard does not intercept outgoing bytes on `ByteSendStream`. It operates on the structured JSON-RPC response object before serialization, avoiding stream adapter complexity.

### Data Model

```python
@dataclass
class ACPWebSocketTransport:
    """Configuration for ACP WebSocket transport (Phase 1).

    Implements the WebSocket subset of the ACP RFC's Streamable HTTP
    WebSocket Transport profile. Phase 1 covers WebSocket only;
    Streamable HTTP (POST/GET/DELETE with SSE) is planned for Phase 2.

    Attributes:
        host: Host to bind the server to.
        port: Port for the server. Note: defaults to 8080 (standard HTTP port).
            This differs from the legacy `WebSocketTransport` default (8765)
            because the transports serve different protocols and use-cases.
            When migrating from legacy `--ws-port`, update to `--port`.
    """
    host: str = "localhost"
    port: int = 8080
```

### Type Union Extension

```python
Transport = (
    StdioTransport
    | WebSocketTransport  # Legacy; deprecated, will be removed
    | StreamTransport
    | ACPWebSocketTransport
    | Literal["stdio", "websocket", "streamable-http"]
)
```

The `"streamable-http"` string literal normalizes to `ACPWebSocketTransport()` with defaults. The `"websocket"` literal continues to normalize to the legacy `WebSocketTransport()` for backward compatibility but emits a deprecation warning.

### API Design

#### WebSocket Upgrade

```
GET /acp HTTP/1.1
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: <key>
Sec-WebSocket-Version: 13

HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: <accept>
Acp-Connection-Id: 550e8400-e29b-41d4-a716-446655440000
```

#### Initialize Lifecycle Flow

```
Client connects via WebSocket upgrade at /acp
        |
        v
  +------------------+
  | initialized=False |
  +------------------+
        |
        v
  Receive first JSON-RPC message
        |
        +---> method == "initialize" ?
        |          |
        |     YES  |         NO
        |          v          v
        |   Pass to       Return JSON-RPC error:
        |   AgentSide     {"jsonrpc":"2.0","error":
        |   Connection      {"code":-32002,
        |                    "message":"initialize required"},
        |                   "id":<request_id>}
        |          |
        |          v
        |   Initialize succeeds?
        |     |
        |  YES |      NO
        |     v        v
        |  Set         Connection remains
        |  initialized  uninitialized;
        |  = True       client may retry
        |     |
        v     v
  Subsequent messages pass through
  to AgentSideConnection unmodified
```

### CLI Integration

```
wolfharness serve-acp config.yml                                      # Default: stdio transport
wolfharness serve-acp config.yml --transport streamable-http          # WebSocket on localhost:8080
wolfharness serve-acp config.yml --transport streamable-http --port 9000   # WebSocket on localhost:9000
wolfharness serve-acp config.yml --transport streamable-http --host 0.0.0.0 --port 9000  # WebSocket on 0.0.0.0:9000
```

The `--transport` flag accepts `stdio` (default), `websocket` (legacy, deprecated), or `streamable-http` (new compliant transport). When `--transport streamable-http` is used, `--host` and `--port` control the bind address. When `--transport websocket` is used, the legacy non-compliant `_serve_websocket()` transport is instantiated; a deprecation warning is emitted directing the user to switch to `--transport streamable-http`. The legacy `--ws-host` and `--ws-port` parameters are still accepted for backward compatibility but emit deprecation warnings directing users to `--host` and `--port`.

### YAML Config Schema Extension

#### ACPPoolServerConfig Fields

The `ACPPoolServerConfig` model in `src/wolfharness_config/pool_server.py` is extended with:

```python
class ACPPoolServerConfig(BasePoolServerConfig):
    # ... existing fields ...

    transport: Literal["stdio", "streamable-http"] = Field(
        default="stdio",
        title="Transport type",
        examples=["stdio", "streamable-http"],
    )
    """Transport type for the ACP server."""

    host: str = Field(
        default="localhost",
        title="Server host",
        examples=["localhost", "0.0.0.0", "127.0.0.1"],
    )
    """Host to bind the server to (streamable-http only)."""

    port: int = Field(
        default=8080,
        gt=0,
        title="Server port",
        examples=[8080, 9000],
    )
    """Port to listen on (streamable-http only)."""
```

#### Example YAML

```yaml
pool_server:
  type: acp
  enabled: true
  transport: streamable-http    # or "stdio" (default)
  host: "0.0.0.0"              # optional, defaults to "localhost"
  port: 8080                    # optional, defaults to 8080
```

When `transport: streamable-http` is specified, `ACPServer.from_config()` resolves the config to an `ACPWebSocketTransport` instance with the provided or default host and port values.

### Uvicorn Shutdown Integration

`ACPServer._start_async()` passes `self._shutdown_event` (an `asyncio.Event` from `BaseServer`) to `serve()`. The `_serve_streamable_http()` function bridges this event to uvicorn's shutdown mechanism:

```python
async def _serve_streamable_http(
    agent: Agent | Callable[[AgentSideConnection], Agent],
    host: str,
    port: int,
    shutdown_event: asyncio.Event | None,
    debug_file: str | None,
    **kwargs: Any,
) -> None:
    from starlette.applications import Starlette
    from starlette.websockets import WebSocket
    from starlette.routing import WebSocketRoute
    import uvicorn

    app = Starlette(routes=[WebSocketRoute("/acp", _acp_websocket_endpoint)])
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)

    shutdown = shutdown_event or asyncio.Event()

    async def shutdown_watcher() -> None:
        await shutdown.wait()
        server.should_exit = True

    watcher_task = asyncio.create_task(shutdown_watcher())

    try:
        await server.serve()
    finally:
        watcher_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher_task
        # Clean up any remaining connections
        # _active_connections: set[AgentSideConnection] maintained by the endpoint handler
        for conn in _active_connections:
            await conn.close()
```

The shutdown sequence is:

1. External signal (SIGTERM) or `ACPServer.stop()` sets `shutdown_event`
2. `shutdown_watcher()` detects the event and sets `server.should_exit = True`
3. uvicorn initiates graceful shutdown, cancelling all in-flight ASGI tasks
4. Each WebSocket endpoint's `try/finally` block calls `AgentSideConnection.close()`
5. `_serve_streamable_http()` awaits uvicorn exit, then cleans up any leaked connections

---

## Security Considerations

### Threat Analysis

| Threat | Impact | Likelihood | Mitigation |
|--------|--------|------------|------------|
| Unauthenticated network access | High | High (if bound to 0.0.0.0) | Default to localhost; document risk of 0.0.0.0 binding |
| WebSocket connection flood (DoS) | Medium | Medium | Add `max_connections` parameter; reject with HTTP 503 when exceeded |
| Unencrypted transport (ws://) | High | Medium | Support TLS via uvicorn flags; recommend reverse proxy for production |
| CORS bypass for future SSE routes | Medium | Low | Add `cors_origins` config parameter; apply Starlette CORSMiddleware |

### Security Measures

- [ ] Default bind to `localhost`; document `0.0.0.0` risk
- [ ] Implement `max_connections` limit with configurable default (100)
- [ ] Support TLS via `--ssl-keyfile`/`--ssl-certfile` CLI flags
- [ ] Add `cors_origins` parameter for Phase 2 SSE routes
- [ ] Log connection ID on connect/disconnect for audit trail

### Compliance

No regulatory requirements apply in Phase 1. Production deployments should run behind a TLS-terminating reverse proxy with authentication middleware.

---

## Implementation Plan

### Phase 1: Transport Config + ASGI WebSocket Server

- **Scope**: New transport type, stream adapters, initialize guard in protocol layer, ASGI app
- **Deliverables**: `ACPWebSocketTransport` dataclass, `_StarletteWebSocketReadStream`/`_StarletteWebSocketWriteStream` adapters, initialize guard in `Connection`, Starlette ASGI app with `/acp` WebSocket route
- **Dependencies**: None (can start immediately)

Tasks:

- 1.1: Add `ACPWebSocketTransport` dataclass to `src/acp/transports.py`
- 1.2: Extend `Transport` type union with `ACPWebSocketTransport` and `"streamable-http"` literal
- 1.3: Add `"streamable-http"` normalization case in `serve()`
- 1.4: Add `ACPWebSocketTransport` dispatch case in `serve()`
- 2.1: Create `_StarletteWebSocketReadStream` adapter (handle `WebSocketDisconnect` -> `anyio.EndOfStream`)
- 2.2: Create `_StarletteWebSocketWriteStream` adapter
- 2.3: Implement initialize lifecycle guard in `Connection._process_message()` or `_agent_handler()`
- 2.4: Create Starlette ASGI app with `/acp` WebSocket route
- 2.5: Implement connection cleanup with `try/finally` in endpoint handler

### Phase 2: Server Runner + ACPServer Integration

- **Scope**: Server runner function, dependency addition, ACPServer integration
- **Deliverables**: `_serve_streamable_http()` function, starlette dependency, verified ACPServer passthrough, uvicorn shutdown bridging
- **Dependencies**: Phase 1 complete

Tasks:

- 3.1: Implement `_serve_streamable_http()` function with uvicorn `Server` direct instantiation
- 3.2: Implement `shutdown_event` -> `server.should_exit` bridging via background watcher task
- 3.3: Add `starlette` dependency to `pyproject.toml`
- 4.1: Verify `ACPServer.__init__()` type hints
- 4.2: Verify `ACPServer.from_config()` type hints
- 4.3: Verify `_start_async()` transport passthrough

### Phase 3: CLI + YAML Config

- **Scope**: CLI flags, YAML schema extension, deprecation of legacy transport
- **Deliverables**: `--transport` extension, `--host`/`--port` flags, YAML `transport: streamable-http` support, legacy `--transport websocket` deprecation warning
- **Dependencies**: Phase 2 complete

Tasks:

- 5.1: Extend `--transport` choices to `Literal["stdio", "websocket", "streamable-http"]`
- 5.2: Add `--host` and `--port` flags (effective when `--transport streamable-http`)
- 5.3: Create `ACPWebSocketTransport` from CLI flags when `--transport streamable-http`
- 5.4: Emit deprecation warning when `--transport websocket` is used (still runs legacy transport, warns user to switch to `--transport streamable-http`)
- 6.1: Extend `ACPPoolServerConfig` with `transport`, `host`, `port` fields following `MCPPoolServerConfig` pattern
- 6.2: Resolve YAML config to `ACPWebSocketTransport` in `ACPServer.from_config()`

### Phase 4: Testing

- **Scope**: Unit and integration tests
- **Deliverables**: Test suite covering all requirements
- **Dependencies**: Phases 1-3 complete

Tasks:

- 7.1: Unit test `ACPWebSocketTransport` dataclass
- 7.2: Unit test `serve()` dispatch for all transport types
- 7.3: Integration test: WebSocket upgrade returns `Acp-Connection-Id`
- 7.4: Integration test: initialize lifecycle enforcement in protocol layer
- 7.5: Integration test: stream adapter round-trip with `WebSocketDisconnect` translation
- 7.6: Integration test: connection cleanup on disconnect
- 7.7: Integration test: graceful shutdown via `shutdown_event`
- 7.8: Integration test: CLI flags and legacy deprecation warning
- 7.9: Integration test: YAML config resolution to `ACPWebSocketTransport`
- 7.10: LSP diagnostics clean on all changed files

### Milestones

| Milestone | Description | Target | Status |
|-----------|-------------|--------|--------|
| M1 | ASGI WebSocket server working end-to-end | Week 1 | Not Started |
| M2 | Server runner + ACPServer integration | Week 1-2 | Not Started |
| M3 | CLI + YAML config complete | Week 2 | Not Started |
| M4 | All tests passing | Week 2-3 | Not Started |

### Rollback Strategy

The new transport is opt-in (requires `--transport streamable-http` or `transport: streamable-http` in YAML). Removing the feature requires:

1. Revert CLI flag additions
2. Revert YAML config model changes
3. Remove `_serve_streamable_http()` and `ACPWebSocketTransport`
4. Remove starlette dependency

No existing functionality is affected since stdio remains the default.

---

## Open Questions

1. **HTTP/2 migration timeline**
   - Context: The RFC may expect HTTP/2. uvicorn supports HTTP/1.1 only. Hypercorn is an ASGI-compatible HTTP/2 server that could replace uvicorn with zero app-code changes.
   - Owner: Team lead
   - Status: Open
2. **Multi-session over single connection design**
   - Context: Phase 1 maps one WebSocket connection to one agent session. The RFC allows multiple sessions over a single connection. The initialize guard's state model may need extension.
   - Owner: Architect
   - Status: Open
3. **Authentication/authorization at the transport layer**
   - Context: Phase 1 has no auth. What mechanisms should Phase 2 support (API key, JWT, mTLS)? Should middleware pipeline be designed now?
   - Owner: Security lead
   - Status: Open
4. **Starlette version pinning strategy**
   - Context: Starlette follows semantic versioning but is developed alongside FastAPI, which can create pressure for rapid minor releases.
   - Owner: DevOps
   - Status: Open

---

## Decision Record

> Complete this section after RFC review is concluded.

### Decision

**Status**: Pending review
**Date**: TBD
**Approvers**: TBD

### Key Discussion Points

1. Framework choice: Starlette vs raw websockets vs FastAPI
2. HTTP/1.1 deviation acceptance
3. Phase 1 scope boundaries (no auth, no multi-session)
4. Initialize enforcement backward compatibility
5. Legacy `WebSocketTransport` deprecation timeline

### Conditions of Approval

- HTTP/1.1 deviation accepted with documented migration path
- starlette dependency audited for transitive dependencies
- Initialize enforcement has opt-out mechanism for migration period
- Legacy `WebSocketTransport` deprecated with removal target v0.6.0 (2026-Q3) and clear migration message to `--transport streamable-http`

### Rejected Options

| Option | Reason |
|--------|--------|
| JSON-RPC error code `-32600` for "initialize required" | **Rejected**: `-32600` is reserved for structural "Invalid Request" errors per JSON-RPC 2.0 spec (§5.1). State-related errors belong in the server error range (`-32000` to `-32099`). `-32002` is idiomatic for server-not-initialized states (also used by LSP). Preserving this record per repository RFC rules. |

### Dissenting Opinions

None recorded yet.

---

## References

### Related Documents

- OpenSpec proposal: `openspec/changes/acp-streamable-http-ws-server/proposal.md`
- OpenSpec design: `openspec/changes/acp-streamable-http-ws-server/design.md`
- OpenSpec ws-transport spec: `openspec/changes/acp-streamable-http-ws-server/specs/ws-transport/spec.md`
- OpenSpec ws-server-integration spec: `openspec/changes/acp-streamable-http-ws-server/specs/ws-server-integration/spec.md`
- OpenSpec tasks: `openspec/changes/acp-streamable-http-ws-server/tasks.md`

### External Resources

- ACP RFC: Streamable HTTP WebSocket Transport profile (`streamable-http-websocket-transport.mdx`)
- Starlette documentation: https://www.starlette.io/
- uvicorn documentation: https://www.uvicorn.org/
- Hypercorn documentation: https://hypercorn.readthedocs.io/

### Appendix

Current transport code: `src/acp/transports.py`
ACP server: `src/wolfharness_server/acp_server/server.py`
Pool server config: `src/wolfharness_config/pool_server.py`
