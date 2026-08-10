# ACP Protocol Gap Analysis: AgentPool vs ACP Spec

**Date:** 2025-05-27  
**AgentPool Commit:** (current working tree)  
**ACP Spec Reference:** `packages/agent-client-protocol/` (Rust reference implementation + docs)  
**Protocol Version:** v1 (stable) + v2 (unstable)

## Executive Summary

AgentPool's ACP implementation covers **most v1 stable features** but has several **naming deviations**, **structural mismatches**, and **missing unstable features** compared to the ACP specification. The implementation is functionally complete for basic agent-client communication but diverges from the spec in ways that may cause interoperability issues with strict ACP clients.

**Key Findings:**
- Several stable methods are incorrectly documented as `UNSTABLE` in AgentPool docstrings (e.g., `session/close`, `session/list`, `session/resume`)
- The `session/fork` capability is fully implemented but can never be advertised due to a missing parameter in `AgentCapabilities.create()`
- `ConfigOptionUpdate` carries extra fields not present in the spec
- `providers/*` methods return untyped raw dictionaries instead of schema-typed responses
- Several draft/unstable features (MCP-over-ACP, NES, `$/cancel_request`) are not implemented

---

## Verification Methodology

This analysis was produced by:

1. **Code-level verification** of AgentPool schema definitions in `packages/wolfharness/src/acp/schema/`
2. **Protocol interface audit** of `packages/wolfharness/src/acp/agent/protocol.py` and `packages/wolfharness/src/acp/client/protocol.py`
3. **Server implementation review** of `packages/wolfharness/src/wolfharness_server/acp_server/`
4. **Cross-reference** with ACP spec Rust types in `packages/agent-client-protocol/src/v1/` and `src/v2/`
5. **Documentation review** of ACP protocol docs in `packages/agent-client-protocol/docs/`

All claims below include specific file paths and line numbers for traceability.

---

## Gap Classification

### P0 — Blocking Protocol Compatibility

*No P0 gaps identified. Wire protocol compatibility is preserved for all implemented features.*

---

### P1 — Structural Alignment

#### 1. `session/close` Named `session/stop` in Implementation

**Status:** Confirmed deviation  
**Impact:** Wire protocol compatible (bridge maps `session/close` → `stop_session`), but internal naming inconsistent

| Spec | Implementation |
|------|---------------|
| Method: `session/close` | Method in protocol: `stop_session` |
| Request: `CloseSessionRequest` | Request: `StopSessionRequest` (`client_requests.py:267`) |
| Response: `CloseSessionResponse` | Response: `StopSessionResponse` (`agent_responses.py:245`) |
| Capability: `session/close` | Capability: `session/close` (advertised correctly) |

**Evidence:**

- `src/acp/schema/messages.py:34` — `AgentMethod` includes `"session/close"` in the Literal
- `src/acp/agent/protocol.py:57` — Interface method is `stop_session()`
- `src/wolfharness_server/acp_server/acp_agent.py:843` — Implementation is `stop_session()`
- `src/acp/bridge/bridge.py:143-146` — Bridge correctly maps `"session/close"` → `stop_session()`

**Note:** The bridge handles the wire protocol mapping, so external clients using `session/close` work correctly. However, internal code and type names use `stop`, which is confusing and inconsistent with the spec.

---

#### 2. `session/fork` Capability Never Advertised

**Status:** Implemented but invisible to clients  
**Impact:** Clients cannot discover that `session/fork` is supported

AgentPool implements `fork_session()` in `acp_agent.py:701` and has `ForkSessionResponse`, but `AgentCapabilities.create()` has **no `fork_session` parameter** and `SessionCapabilities` construction never includes a `fork` field:

```python
# src/acp/schema/capabilities.py:307-311
session_caps = SessionCapabilities(
    list=SessionListCapabilities() if list_sessions else None,
    resume=SessionResumeCapabilities() if resume_session else None,
    close=SessionCloseCapabilities() if close_session else None,
    # fork is missing!
)
```

This means the capability can never be advertised to clients, despite the method being fully implemented.

---

#### 3. `providers/*` Implemented as Extension Methods with Untyped Responses

**Status:** Functionally complete, but not native ACP methods  
**Impact:** Works with AgentPool's bridge, but responses are untyped raw dicts

The spec defines `providers/list`, `providers/set`, `providers/disable` as **native agent methods** with typed responses (`ListProvidersResponse`, `SetProvidersResponse`, `DisableProvidersResponse`). AgentPool implements them in `ext_method()` returning raw dicts:

```python
# src/wolfharness_server/acp_server/acp_agent.py:875-898
async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
    match method:
        case "providers/list":
            return {"providers": [...]}   # ← untyped
        case "providers/set":
            return {"success": True}      # ← untyped
        case "providers/disable":
            return {"success": True}      # ← untyped
```

**Missing from schema:** No `ProvidersListRequest`, `ProvidersListResponse`, etc. in `ClientRequest` / `AgentResponse` union types.

**Evidence of partial schema support:**
- `src/acp/schema/providers.py` — Defines `ProviderInfo`, `ProviderStatus`
- `src/acp/schema/capabilities.py:272` — `providers: bool` capability flag
- `src/wolfharness_server/acp_server/provider_router.py` — Full provider management logic

---

#### 4. `ToolCallLocation.line` Defaults to `0` Instead of Optional

**Status:** Confirmed deviation  
**Impact:** Semantic difference — `0` means "beginning/unspecified" instead of "not provided"

```python
# src/acp/schema/tool_call.py:298-310
class ToolCallLocation(AnnotatedObject):
    line: int = Field(default=0, ge=0)  # ← default=0, should be Optional
    path: str
```

The spec treats `line` as optional (omitted when not applicable). AgentPool defaults to `0`, which is semantically different from "not specified" and may confuse clients doing `if location.line is not None` checks.

---

### P2 — Spec Deviations (Non-Blocking)

#### 5. `ConfigOptionUpdate` Contains Extra Fields

**Status:** Confirmed deviation  
**Impact:** Protocol compatible (extra fields are ignored by lenient JSON parsers), but violates spec minimalism

The spec requires `ConfigOptionUpdate` to contain **only**:
- `session_update: "config_option_update"`
- `config_options: SessionConfigOption[]`

AgentPool adds two extra fields:

```python
# src/acp/schema/session_updates.py:380-397
class ConfigOptionUpdate(AnnotatedObject):
    session_update: Literal["config_option_update"] = ...
    config_id: str          # ← EXTRA: not in spec
    value_id: str           # ← EXTRA: not in spec
    config_options: Sequence[SessionConfigOption]  # ← spec only requires this
```

**Spec reference:** `packages/agent-client-protocol/docs/protocol/session-config-options.mdx` states the update "contains the new set of `configOptions`" with no mention of `config_id` or `value_id`.

---

#### 6. Stable Methods Incorrectly Marked as `UNSTABLE`

**Status:** Confirmed documentation error  
**Impact:** Misleading for developers; no runtime impact

The Rust spec shows these methods are **stable** (not behind `cfg` flags), but AgentPool's docstrings incorrectly mark them as unstable:

| Method | Spec Status | AgentPool Docstring |
|--------|-------------|-------------------|
| `session/close` | Stable | `StopSessionRequest` docstring says "UNSTABLE" |
| `session/list` | Stable | `ListSessionsRequest` docstring says "UNSTABLE" |
| `session/resume` | Stable | `ResumeSessionRequest` docstring says "UNSTABLE" |

Note: `session/fork` **is** unstable in the spec (behind `cfg`), so marking it unstable is correct.

---

#### 7. HTTP/2 Not Supported for Streamable HTTP

**Status:** Confirmed — uses HTTP/1.1  
**Impact:** ACP RFD draft requires HTTP/2 for streamable HTTP transport

```python
# src/acp/bridge/bridge.py:188-205
# Uses Starlette + uvicorn with no HTTP/2 configuration
config = uvicorn.Config(app, host=..., port=..., log_level=...)
```

The spec RFD (`docs/rfds/streamable-http-websocket-transport.mdx`) explicitly states:
> "The transport layer MUST support HTTP/2"

AgentPool's bridge and server implementations use standard HTTP/1.1 via Starlette/uvicorn. This applies to the **draft RFD transport**, not the stable v1 core protocol.

---

#### 8. Dual-Header Identity Model Not Implemented

**Status:** Confirmed missing  
**Impact:** Limits session recovery and connection resilience

The spec RFD defines:
- `Acp-Connection-Id` — Identifies the persistent connection
- `Acp-Session-Id` — Identifies the session within that connection

AgentPool does not implement these headers in either the bridge or the server transport. This applies to the **draft RFD transport**, not the stable v1 core protocol.

---

### P3 — Feature Gaps

#### 9. `$/cancel_request` (Protocol-Level Cancellation) Not Implemented

**Status:** Confirmed missing  
**Impact:** Cannot cancel pending requests at JSON-RPC level

The spec defines `$/cancel_request` as a protocol-level method to cancel in-flight requests (returns `-32800` error to original request). AgentPool only implements `session/cancel` (session-level cancellation via `CancelNotification`).

**Evidence:**
- `src/acp/schema/messages.py:22-35` — `AgentMethod` does not include `$/cancel_request`
- `src/acp/schema/notifications.py:32-47` — Only `CancelNotification` (session-level) exists

---

#### 10. MCP-over-ACP Not Implemented

**Status:** Confirmed missing  
**Impact:** AgentPool cannot act as MCP proxy over ACP

The spec defines `mcp/connect`, `mcp/message`, `mcp/disconnect` for tunneling MCP through ACP. AgentPool has MCP server support (`src/wolfharness/mcp_server/`) but no MCP-over-ACP bridge.

**Evidence:**
- `src/acp/schema/messages.py:22-35` — No `mcp/*` methods in `AgentMethod`
- `src/acp/schema/capabilities.py:151-158` — `McpCapabilities` only covers HTTP/SSE, not `acp` transport
- `src/acp/bridge/bridge.py:876` — Comment mentions "MCP-over-ACP lifecycle messages" but no implementation

---

#### 11. NES (Next Edit Suggestions) Not Implemented

**Status:** Confirmed missing  
**Impact:** No predictive editing support

Spec methods: `nes/start`, `nes/suggest`, `nes/accept`, `nes/reject`, `nes/close`

**Evidence:** No references to NES methods in any schema or server file.

---

#### 12. `session/delete` Not Implemented

**Status:** Confirmed missing  
**Impact:** Cannot permanently delete sessions

The spec RFD defines `session/delete` (draft stage) for permanent session removal. AgentPool has `session/close` (mapped to `stop_session`) which frees resources but does not delete persisted session data.

---

#### 13. `logout` Not Implemented

**Status:** Confirmed missing  
**Impact:** Cannot clear authentication state

The spec defines `logout` (unstable) for clearing authentication tokens. AgentPool implements `authenticate` but not `logout`.

---

### P4 — Forward Compatibility Risks

#### 14. Closed `AgentMethod` Literal Blocks Future Methods

**Status:** Confirmed risk  
**Impact:** New spec methods or extension methods will cause deserialization failures

`AgentMethod` is a closed `Literal` with no fallback mechanism:

```python
# src/acp/schema/messages.py:22-35
AgentMethod = Literal[
    "authenticate", "initialize", "session/cancel", ...
]
```

If a client sends any method not in this list (e.g., a future spec method or an extension method without `_` prefix), Pydantic validation in `AgentRequestMessage` will reject it at the schema level. The spec's Rust types use `#[non_exhaustive]` enums and string fallback patterns for forward compatibility. AgentPool has no such mechanism.

This is more severe than generic enum fallback because it affects **method dispatch** — the entry point of all agent requests.

---

#### 15. No Fallback for Unknown `SessionUpdate` Types

**Status:** Confirmed risk  
**Impact:** New session update types will break client-side parsing

The `SessionUpdate` union in `session_updates.py` has no fallback variant. If an agent sends a new update type not in the discriminated union, client-side Pydantic parsing will fail.

---

#### 16. No Enum Fallback for Discriminated Unions

**Status:** Confirmed risk  
**Impact:** New spec variants in other enums will cause deserialization failures

Rust spec uses `#[serde(other)]` for enum fallback:

```rust
// Rust spec pattern
enum SomeType {
    KnownVariant,
    #[serde(other)]
    Unknown,
}
```

AgentPool uses Python `Literal` and `StrEnum` with **no fallback** in most places:

```python
# src/acp/schema/providers.py:19-26
class ProviderStatus(StrEnum):
    enabled = "enabled"
    disabled = "disabled"
```

If a spec update adds new status values, AgentPool will fail to parse them instead of gracefully handling unknown values.

---

## Correctly Implemented Features

The following features are verified to be **fully compliant** with the ACP spec:

| Feature | Status | Evidence |
|---------|--------|----------|
| `initialize` | ✅ | `src/acp/schema/client_requests.py:195`, `agent_responses.py:256` |
| `authenticate` | ✅ | `src/acp/schema/client_requests.py:283`, `agent_responses.py:252` |
| `session/new` | ✅ | `src/acp/schema/client_requests.py:31`, `agent_responses.py:47` |
| `session/load` | ✅ | `src/acp/schema/client_requests.py:44`, `agent_responses.py:82` |
| `session/prompt` | ✅ | `src/acp/schema/client_requests.py:129`, `agent_responses.py:223` |
| `session/cancel` | ✅ | `src/acp/schema/notifications.py:32` |
| `session/set_mode` | ✅ | `src/acp/schema/client_requests.py:119`, `agent_responses.py:212` |
| `session/set_model` | ✅ | `src/acp/schema/client_requests.py:166`, `agent_responses.py:40` |
| `session/set_config_option` | ✅ | `src/acp/schema/client_requests.py:179`, `agent_responses.py:216` |
| `session/list` | ✅ | `src/acp/schema/client_requests.py:62`, `agent_responses.py:341` |
| `session/fork` | ⚠️ | Implemented (`client_requests.py:77`, `agent_responses.py:103`) but **capability cannot be advertised** (see Gap #2) |
| `session/resume` | ✅ | `src/acp/schema/client_requests.py:98`, `agent_responses.py:160` |
| `fs/read_text_file` | ✅ | `src/acp/schema/agent_requests.py:33`, `client_responses.py:56` |
| `fs/write_text_file` | ✅ | `src/acp/schema/agent_requests.py:20`, `client_responses.py:57` |
| `terminal/*` | ✅ | `src/acp/schema/agent_requests.py:49-100`, `client_responses.py:27-53` |
| `session/request_permission` | ✅ | `src/acp/schema/agent_requests.py:103`, `client_responses.py:58` |
| `elicitation/create` | ✅ | `src/acp/schema/elicitation.py:12`, `agent_requests.py:122` |
| `elicitation/complete` | ✅ | `src/acp/schema/elicitation.py:70`, `client/protocol.py:42` |
| `session/update` notifications | ✅ | `src/acp/schema/session_updates.py` (all types) |
| `_meta` propagation | ✅ | `src/acp/schema/base.py:11-31` |
| Capability negotiation | ✅ | `src/acp/schema/capabilities.py` |
| Tool call lifecycle | ✅ | `src/acp/schema/tool_call.py` |
| JSON-RPC 2.0 framing | ✅ | `src/acp/schema/messages.py:51` |
| Extension methods (`_` prefix) | ✅ | `src/acp/agent/protocol.py:73`, `client/protocol.py:68` |

---

## AgentPool-Specific Extensions (Beyond ACP Spec)

These features are **not in the ACP spec** but are intentional AgentPool extensions:

| Extension | Purpose | File |
|-----------|---------|------|
| Session-scoped interrupt routing | Cross-task interrupt routing via `session_id` registry | RFC-0023 |
| Per-session agent isolation | Independent agent instances per ACP session | RFC-0031 |
| Zed subagent `_meta` extensions | `subagent_session_info`, `tool_name` in `_meta` | RFC-0027 |
| Session-scoped MCP isolation | `{server_name}_` prefix for session-level tools | `openspec/per-session-mcp-isolation` |

---

## Recommended Fix Priority

### Immediate (P1)

1. **Fix `session/fork` capability advertising** — Add `fork_session` parameter to `AgentCapabilities.create()` and `fork` field to `SessionCapabilities`
2. **Rename `session/stop` → `session/close`** across all internal types and method names while keeping backward-compatible bridge mapping
3. **Add typed `providers/*` request/response schemas** and register them in native method dispatch instead of `ext_method()`
4. **Make `ToolCallLocation.line` optional** (`int | None = None`)

### Short-term (P2)

5. **Remove `config_id` and `value_id`** from `ConfigOptionUpdate`, keeping only `config_options`
6. **Fix incorrect `UNSTABLE` docstrings** on `session/close`, `session/list`, `session/resume` request types
7. **Add HTTP/2 support** in bridge/server for RFD draft transport (via uvicorn HTTP/2 config or hyper)
8. **Implement `Acp-Connection-Id` / `Acp-Session-Id` headers** for RFD draft transport

### Medium-term (P3)

9. **Implement `$/cancel_request`** protocol-level cancellation
10. **Implement `session/delete`** for permanent session removal
11. **Add enum fallback handling** for `AgentMethod`, `SessionUpdate`, and other discriminated unions

### Lower Priority (P4)

12. **Implement MCP-over-ACP** if MCP proxy use case is needed
13. **Implement NES** if predictive editing is a product requirement
14. **Implement `logout`** for auth state cleanup
15. **Restructure `WaitForTerminalExitResponse`** to use nested `TerminalExitStatus` (internal style only — no wire impact due to `#[serde(flatten)]`)

---

## Appendix: Oracle Review Notes

This document was reviewed by an Oracle agent with full codebase access. Key corrections applied:

- **Gap #1** (`session/close` vs `stop`) downgraded from P0 → P1 because wire protocol compatibility is preserved via bridge mapping
- **Gap #2** (`ConfigOptionUpdate` extra fields) downgraded from P0 → P2 because extra JSON fields are ignored by lenient parsers
- **Gap #3** (`WaitForTerminalExitResponse`) removed as a wire-protocol gap because Rust spec uses `#[serde(flatten)]`, producing identical JSON on the wire
- **Gap #14** (`message_id` unstable marking) removed because the spec also marks `message_id` as unstable (behind `cfg` flag)
- **New Gap #2** added: `session/fork` capability can never be advertised due to missing factory parameter
- **New Gap #6** added: Stable methods (`session/close`, `session/list`, `session/resume`) incorrectly documented as `UNSTABLE`
- **New Gap #14** added: Closed `AgentMethod` Literal blocks future spec methods and extension methods
- **New Gap #15** added: No fallback for unknown `SessionUpdate` types
- **Correctly Implemented** table updated with caveat on `session/fork`

---

## Appendix: File Index

Key files referenced in this analysis:

**Schema Definitions:**
- `packages/wolfharness/src/acp/schema/base.py` — Base models, `_meta` handling
- `packages/wolfharness/src/acp/schema/messages.py` — `AgentMethod`, `ClientMethod` enums
- `packages/wolfharness/src/acp/schema/client_requests.py` — All client→agent request types
- `packages/wolfharness/src/acp/schema/agent_responses.py` — All agent→client response types
- `packages/wolfharness/src/acp/schema/session_updates.py` — Session update notification types
- `packages/wolfharness/src/acp/schema/tool_call.py` — Tool call types, `ToolCallLocation`
- `packages/wolfharness/src/acp/schema/client_responses.py` — Client response types, terminal
- `packages/wolfharness/src/acp/schema/elicitation.py` — Elicitation types
- `packages/wolfharness/src/acp/schema/providers.py` — Provider types
- `packages/wolfharness/src/acp/schema/capabilities.py` — Capability flags
- `packages/wolfharness/src/acp/schema/notifications.py` — Notification types

**Protocol Interfaces:**
- `packages/wolfharness/src/acp/agent/protocol.py` — Agent-side protocol interface
- `packages/wolfharness/src/acp/client/protocol.py` — Client-side protocol interface

**Server Implementation:**
- `packages/wolfharness/src/wolfharness_server/acp_server/acp_agent.py` — ACP agent protocol implementation
- `packages/wolfharness/src/wolfharness_server/acp_server/provider_router.py` — Provider management
- `packages/wolfharness/src/acp/bridge/bridge.py` — ACP bridge (stdio → HTTP)

**ACP Spec Reference:**
- `packages/agent-client-protocol/src/v1/` — Rust v1 stable types
- `packages/agent-client-protocol/src/v2/` — Rust v2 unstable types
- `packages/agent-client-protocol/docs/protocol/` — Protocol documentation
- `packages/agent-client-protocol/docs/rfds/` — Request for Discussion documents
