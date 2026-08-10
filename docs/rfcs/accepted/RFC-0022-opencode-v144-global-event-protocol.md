---
rfc_id: RFC-0022
title: OpenCode v1.4.4+ GlobalEvent Protocol Support
status: ACCEPTED
author: yuchen.liu
reviewers: []
created: 2026-04-15
last_updated: 2026-04-16
decision_date: 2026-04-16
related_prds: []
related_rfcs:
  - RFC-0013-subagent-event-unification.md
  - RFC-0014-spawn-session-events.md
---

# RFC-0022: OpenCode v1.4.4+ GlobalEvent Protocol Support

## Overview

OpenCode v1.4.4 introduced a breaking change to the SSE event protocol. The TUI now subscribes to `/global/event` (instead of `/event`) and expects events in a new `GlobalEvent` envelope format with `directory` routing field. Without this field, the TUI cannot route events to the correct project context and silently drops all SSE events, resulting in a completely non-functional UI despite the server processing messages successfully.

This RFC proposes updating wolfharness's `/global/event` SSE output to conform to the new `GlobalEvent` protocol, ensuring compatibility with OpenCode v1.4.4+ while maintaining backward compatibility with v1.4.3 clients via the unchanged `/event` endpoint.

> **Review Status**: This RFC has been reviewed by Oracle and Metis against the actual OpenCode v1.4.4 source at `~/src/opencode`. Key corrections applied: (1) `directory` must use `working_dir` not `base_path` to match the `/path` endpoint, (2) `workspace` field is optional in the SDK and should be omitted for single-directory servers, (3) `server.connected`/`server.heartbeat` must NOT be wrapped in GlobalEvent, (4) SyncEvent support is unnecessary for basic TUI functionality.

## Table of Contents

- Background & Context
- [Problem Statement](#problem-statement)
- Goals & Non-Goals
- [Evaluation Criteria](#evaluation-criteria)
- [Options Analysis](#options-analysis)
- [Recommendation](#recommendation)
- [Technical Design](#technical-design)
- Security Considerations
- Implementation Plan
- [Open Questions](#open-questions)
- Decision Record
- [References](#references)

---

## Background & Context

### Current State

AgentPool's OpenCode server exposes two SSE endpoints:

1. **`/event`** — raw event stream (no wrapper), used by OpenCode v1.4.3 and earlier
2. **`/global/event`** — payload-wrapped event stream, used by OpenCode v1.4.4+

The current `_serialize_event()` in `global_routes.py` produces:

```json
{"payload": {"type": "message.part.updated", "properties": {...}}}
```

### Historical Context

OpenCode v1.4.4 (commits `b22add292`, `42206da1f`) introduced the following breaking changes:

1. **SSE endpoint migration**: TUI subscribes to `/global/event` instead of `/event`
2. **GlobalEvent envelope**: Events must include `directory` routing field; `project` and `workspace` are optional
3. **Unified event emitter**: SDK now uses `emitter.emit("event", event)` instead of `emitter.emit(event.type, event)`
4. **Sync event format**: Sync events use `type: "sync"` — but TUI discards them in `useEvent()`, so not needed for external servers

### Glossary

| Term | Definition |
|------|------------|
| GlobalEvent | OpenCode v1.4.4+ event envelope with `directory`, optional `project`/`workspace`, and `payload` fields |
| SyncEvent | Event with `type: "sync"` — used internally by OpenCode for event sourcing; TUI discards them in `useEvent()` |
| Directory routing | TUI routes events by matching `event.directory` against `project.instance.directory()` |
| Workspace routing | TUI routes events by matching `event.workspace` against `project.workspace.current()`; takes priority over directory routing when a workspace is active |
| Global directory | `directory: "global"` is a special value that causes events to be delivered regardless of project context |

### OpenCode v1.4.4 TUI Event Routing Logic

From `packages/opencode/src/cli/cmd/tui/context/event.ts` (verified against source):

```typescript
// useEvent() routing logic (exact):
// 1. Global events (always delivered)
if (event.directory === "global") { handler(event.payload) }
// 2. Sync events (discarded by useEvent)
if (event.payload.type === "sync") return;
// 3. Workspace routing (takes priority, BLOCKS directory check)
if (project.workspace.current()) {
  if (event.workspace === project.workspace.current()) { handler(event.payload) }
  return  // ← directory check is SKIPPED when workspace is active
}
// 4. Directory routing (fallback when no workspace active)
if (event.directory === project.instance.directory()) { handler(event.payload) }
```

**Critical**: If `directory` is missing from the event, the TUI cannot match it and silently drops the event. The `directory` value MUST match what the `/path` API endpoint returns (i.e., `state.working_dir`), NOT the resolved path (`state.base_path`), because the TUI's `project.instance.directory()` is initialized from the `/path` response.

---

## Problem Statement

### The Problem

AgentPool's `/global/event` endpoint produces events that lack the `directory` routing field required by OpenCode v1.4.4+ TUI. As a result:

- The TUI connects to the SSE stream successfully
- The server broadcasts events (visible in server logs as `SSE: Sending event`)
- The TUI receives events but cannot route them (no `directory` match)
- **All agent responses are silently dropped** — the UI appears frozen

### Evidence

- Debug logs confirm: server processes messages, LLM calls succeed (HTTP 200), SSE events are broadcast
- TUI shows no response after message submission
- Server logs show `SSE: Sending event` for `session.status`, `message.part.updated`, `message.part.delta`
- Current output: `{"payload": {"type": "message.part.updated", ...}}` — missing `directory`, `project`, `workspace`
- Required output: `{"directory": "/path/to/project", "project": "xxx", "workspace": "yyy", "payload": {...}}`

### Impact of Inaction

- **Critical**: All users on OpenCode v1.4.4+ cannot use wolfharness's `serve-opencode` at all
- **Blocking**: wolfharness is effectively incompatible with the current OpenCode release
- **Risk**: As OpenCode auto-updates, the user base on v1.4.4+ will only grow
- **No workaround**: Pinning to v1.4.3 is not sustainable (security updates, bug fixes)

---

## Goals & Non-Goals

### Goals (In Scope)

1. Add `directory` routing field to `/global/event` SSE output (required for TUI event delivery)
2. Add optional `project` field for informational purposes
3. Maintain backward compatibility via the unchanged `/event` endpoint for v1.4.3 clients
4. Derive routing fields from existing `ServerState` (`working_dir`, computed `project_id`)
5. Do NOT wrap `server.connected`/`server.heartbeat` in GlobalEvent (matches OpenCode's own server behavior)
6. Pass manual verification: messages appear in OpenCode v1.4.4+ TUI after submission

### Non-Goals (Out of Scope)

1. Rewriting the entire event system — only the serialization layer changes
2. Supporting the OpenCode Desktop app (`packages/app`) — uses same protocol but different SDK entry point
3. Adding workspace-level isolation (multi-workspace) — wolfharness serves a single directory; `workspace` field omitted
4. Protocol versioning or negotiation — not needed for this change
5. Changing the `/event` endpoint behavior (backward compat preserved by default)
6. SyncEvent emission — TUI discards sync events in `useEvent()`; not needed for basic TUI functionality

### Success Criteria

- [ ] OpenCode v1.4.4+ TUI displays agent responses after message submission
- [ ] SSE events include `directory` field matching the `/path` endpoint's `directory` value
- [ ] `/event` endpoint still works for OpenCode v1.4.3 (no regression)
- [ ] `server.connected` and `server.heartbeat` emitted without GlobalEvent wrapper
- [ ] No new mypy or ruff errors

---

## Evaluation Criteria

| Criterion | Weight | Description | Minimum Threshold |
|-----------|--------|-------------|-------------------|
| Protocol correctness | High | Events match OpenCode v1.4.4+ GlobalEvent schema exactly | All required fields present |
| Backward compatibility | High | v1.4.3 clients continue to work via `/event` | Zero regression |
| Implementation simplicity | Medium | Minimal code changes, no architectural refactoring | No new dependencies |
| Runtime overhead | Low | No measurable performance impact on SSE throughput | <1ms per event |
| Maintainability | Medium | Easy to update when OpenCode protocol changes again | Centralized serialization |

---

## Options Analysis

### Option 1: Minimal GlobalEvent Envelope in `_serialize_event()`

**Description**

Add `directory`, `project`, and `workspace` fields directly in the `_serialize_event()` function in `global_routes.py`. Pass `ServerState` (or just the needed fields) into the serialization function. When `wrap_payload=True`, construct the full GlobalEvent envelope. Leave `_serialize_event()` with `wrap_payload=False` unchanged for the `/event` endpoint.

**Advantages**

- Minimal code change — one function modification
- No new models or data structures required
- Centralized change — easy to review and revert
- `ServerState` already has `working_dir` and `base_path` properties

**Disadvantages**

- `_serialize_event()` gains awareness of server state (currently pure function)
- `project_id` computation happens on every event (though lightweight)
- `workspace` field is intentionally omitted — wolfharness doesn't have a workspace concept

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Protocol correctness | High | All fields can be derived from ServerState |
| Backward compatibility | High | `/event` unchanged, `/global/event` gets new fields |
| Implementation simplicity | High | Single function change |
| Runtime overhead | High | Negligible — string operations |
| Maintainability | Medium | Logic in route file, not in model layer |

**Effort Estimate**

- Complexity: Low
- Resources: 1 engineer, <1 day
- Dependencies: None

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `workspace` field semantics unclear | Medium | Low | Omit `workspace` entirely; document that wolfharness doesn't support workspaces |
| OpenCode changes protocol again | Low | Medium | Centralized serialization makes updates easy |

---

### Option 2: Formal GlobalEvent Model with Dedicated Serializer

**Description**

Create a `GlobalEvent` Pydantic model in `models/events.py` that wraps any `Event` with `directory`, `project`, `workspace`, and `payload` fields. Create a `SyncEvent` model for the sync envelope. Move serialization logic to a dedicated module or class that takes `ServerState` as context.

**Advantages**

- Type-safe event construction — Pydantic validates the envelope
- Models serve as living documentation of the protocol
- Easier to extend when protocol changes (add fields to the model)
- Clear separation: event models vs. route logic
- SyncEvent gets proper modeling

**Disadvantages**

- More code changes (new models, serialization refactor)
- Slightly higher implementation effort
- Need to decide where the `GlobalEvent` construction happens (serializer class? route handler?)

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Protocol correctness | High | Pydantic validation ensures correct schema |
| Backward compatibility | High | `/event` endpoint unchanged |
| Implementation simplicity | Medium | More models and wiring, but straightforward |
| Runtime overhead | High | Pydantic serialization is fast |
| Maintainability | High | Model-driven, easy to evolve |

**Effort Estimate**

- Complexity: Medium
- Resources: 1 engineer, 1-2 days
- Dependencies: None

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Over-engineering for current needs | Low | Low | Models are small and focused |
| Pydantic validation overhead | Very Low | Very Low | Models are simple wrappers |

---

### Option 3: Protocol Versioning with Dual Serialization

**Description**

Add a protocol version header or query parameter to the SSE endpoints. Serialize events differently based on the client's declared protocol version. This would allow gradual migration and explicit version negotiation.

**Advantages**

- Explicit versioning — client declares what it supports
- Can support both v1.4.3 and v1.4.4+ simultaneously with same endpoint
- Future-proof for more protocol changes

**Disadvantages**

- Significantly more complex — need version detection logic
- OpenCode doesn't currently send a version header
- Would require OpenCode-side changes for version negotiation
- Over-engineered for a one-time breaking change
- Both endpoints already exist (`/event` vs `/global/event`) — natural version split

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Protocol correctness | High | Version-specific serialization |
| Backward compatibility | High | Explicit versioning |
| Implementation simplicity | Low | Complex version detection and dual paths |
| Runtime overhead | Medium | Version check on every event |
| Maintainability | Medium | Multiple serialization paths to maintain |

**Effort Estimate**

- Complexity: High
- Resources: 1 engineer, 3-5 days
- Dependencies: May require OpenCode client changes

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| OpenCode doesn't support version negotiation | High | High | Would need to fall back to Option 1 or 2 |
| Maintenance burden of dual paths | Medium | Medium | Keep paths as thin as possible |

---

### Options Comparison Summary

| Criterion | Option 1: Minimal Envelope | Option 2: Formal GlobalEvent Model | Option 3: Protocol Versioning |
|-----------|---------------------------|-----------------------------------|-------------------------------|
| Protocol correctness | High | High | High |
| Backward compatibility | High | High | High |
| Implementation simplicity | High | Medium | Low |
| Runtime overhead | High | High | Medium |
| Maintainability | Medium | High | Medium |
| **Overall** | **High** | **High** | **Low** |

---

## Recommendation

### Recommended Option

**Option 2: Formal GlobalEvent Model with Dedicated Serializer**

### Justification

While Option 1 is the quickest path, the GlobalEvent envelope is a protocol-level concern that deserves proper modeling. OpenCode's protocol is likely to evolve further (the v1.4.4 change demonstrates this). A Pydantic model provides:

1. **Type safety**: Invalid events caught at construction time, not at runtime
2. **Documentation**: The model IS the protocol spec
3. **Evolvability**: Adding fields to `GlobalEvent` or `SyncEvent` is a model change, not a logic change
4. **Low incremental cost**: The extra effort over Option 1 is minimal (2 small models + wiring)

Option 3 is rejected because the natural version split already exists via separate endpoints (`/event` for v1.4.3, `/global/event` for v1.4.4+). Adding explicit version negotiation would be over-engineering.

### Accepted Trade-offs

1. **More code than Option 1**: Acceptable because the `GlobalEvent` model is small (~40 lines with docstrings) and provides type safety benefits that compound over time.
2. **Omitting `workspace` field**: AgentPool doesn't have workspaces. If workspace support is added later, the `workspace` field can be populated at that time. For now, omitting it is the correct choice because `workspace=project_id` would be factually wrong (WorkspaceID format is `wrk_<hex>`, not a git SHA).
3. **Using `working_dir` instead of `base_path` for directory**: The `/path` endpoint uses `working_dir`, so the GlobalEvent must use the same value for consistency. This means symlinked paths won't be resolved, but this matches the current `/path` behavior.

### Conditions

- Must not break the existing `/event` endpoint
- Must pass manual testing with OpenCode v1.4.4+ TUI
- Must add/update type annotations passing mypy strict mode

---

## Technical Design

### Architecture Overview

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  ServerState │────▶│ GlobalEventFactory│────▶│  SSE Response    │
│  (context)   │     │  (serializer)     │     │  /global/event   │
└─────────────┘     └──────────────────┘     └─────────────────┘
                             │
                      ┌──────┴──────┐
                      │  GlobalEvent │  (Pydantic model)
                      └─────────────┘
```

### Key Components

#### 1. GlobalEvent Model

New model in `models/events.py`:

```python
class GlobalEvent(OpenCodeBaseModel):
    """GlobalEvent envelope for OpenCode v1.4.4+ SSE protocol.
    
    Wraps an Event with routing fields that allow the TUI to
    dispatch events to the correct project context.
    
    Based on OpenCode SDK types (types.gen.ts lines 1097-1100):
    - directory: required, used for TUI event routing
    - project: optional, informational (git root commit SHA)
    - workspace: optional, only needed when workspaces are active
    - payload: the actual event data
    """

    directory: str
    """Working directory — MUST match the /path endpoint's directory value.
    
    The TUI routes events by comparing event.directory against
    project.instance.directory(), which comes from the /path API response.
    Using state.working_dir (not state.base_path) ensures consistency
    with the /path endpoint and avoids symlink resolution mismatches.
    """

    project: str | None = None
    """Project identifier — computed from working directory via helpers.compute_project_id().
    
    Optional in the SDK type. Included for informational purposes.
    Value is the git root commit SHA (or "global" for non-git dirs).
    """

    workspace: str | None = None
    """Workspace identifier — OMITTED for single-directory servers.
    
    In OpenCode, workspace IDs have format 'wrk_<hex><random>' — they
    are NOT the same as project_id (which is a git SHA). Setting
    workspace=project_id is INCORRECT and would break routing if
    workspaces are ever enabled.
    
    When no workspace is active, the TUI falls through to directory-based
    routing, so omitting this field is safe and correct.
    """

    payload: dict[str, Any]
    """The actual event data (serialized Event)."""
```

#### 2. GlobalEventFactory

New class stored on `ServerState` (not per-connection, since fields are immutable):

```python
class GlobalEventFactory:
    """Creates GlobalEvent envelopes from Event instances using ServerState context.
    
    Stored on ServerState since directory/project don't change during
    the server's lifetime. Created lazily on first access.
    """

    def __init__(self, state: ServerState) -> None:
        # CRITICAL: Use working_dir, NOT base_path.
        # The /path endpoint returns working_dir, and the TUI's
        # project.instance.directory() comes from the /path response.
        # If base_path (which resolves symlinks) is used instead,
        # symlinked paths won't match and events get dropped.
        self._directory = state.working_dir
        self._project = helpers.compute_project_id(state.working_dir)
        # workspace is intentionally omitted (None) — wolfharness
        # doesn't have workspaces, and workspace=project_id would
        # be wrong (WorkspaceID format is 'wrk_xxx', not a git SHA)

    def wrap(self, event: Event) -> GlobalEvent:
        """Wrap an Event in a GlobalEvent envelope."""
        event_data = event.model_dump(by_alias=True, exclude_none=True)

        # Add sessionId at top level of payload if available
        session_id = _extract_session_id(event)
        if session_id is not None:
            event_data["sessionId"] = session_id

        return GlobalEvent(
            directory=self._directory,
            project=self._project,
            # workspace omitted (None → excluded by exclude_none=True)
            payload=event_data,
        )

    @staticmethod
    def is_global_only_event(event: Event) -> bool:
        """Check if an event should be emitted without GlobalEvent wrapper.
        
        OpenCode's own server (global.ts lines 27-33) emits
        server.connected and server.heartbeat as bare payloads
        without routing fields. These are connection-level events,
        not project-scoped events.
        """
        return isinstance(event, ServerConnectedEvent | ServerHeartbeatEvent)
```

#### 3. Updated `_event_generator`

```python
async def _event_generator(
    state: ServerState, *, wrap_payload: bool = False
) -> AsyncGenerator[dict[str, Any]]:
    factory = state.get_event_factory() if wrap_payload else None
    # ... existing queue setup ...
    try:
        connected = ServerConnectedEvent()
        # server.connected is NOT wrapped in GlobalEvent (matches OpenCode's behavior)
        data = _serialize_event(connected, wrap_payload=False)
        yield {"data": data}

        while True:
            event = await queue.get()
            if factory and not GlobalEventFactory.is_global_only_event(event):
                data = factory.wrap(event).model_dump_json(exclude_none=True)
            else:
                data = _serialize_event(event, wrap_payload=False)
            yield {"data": data}
    finally:
        # ... existing cleanup ...
```

#### 4. ServerState Integration

```python
@dataclass
class ServerState:
    # ... existing fields ...
    _event_factory: GlobalEventFactory | None = field(default=None, repr=False)

    def get_event_factory(self) -> GlobalEventFactory:
        """Get or create the GlobalEvent factory (lazy init)."""
        if self._event_factory is None:
            self._event_factory = GlobalEventFactory(self)
        return self._event_factory
```

### Data Model

#### Before (v1.4.3, `/global/event`):

```json
{"payload": {"type": "message.part.updated", "properties": {"part": {...}}}}
```

#### After (v1.4.4+, `/global/event` — regular events):

```json
{
  "directory": "/Users/dev/my-project",
  "project": "a1b2c3d4",
  "payload": {
    "type": "message.part.updated",
    "sessionId": "sess_abc123",
    "properties": {"part": {...}}
  }
}
```

Note: `workspace` field is omitted for single-directory servers. `project` is optional in the SDK but included for informational purposes.

#### After (v1.4.4+, `/global/event` — server.connected):

```json
{"type": "server.connected", "properties": {}}
```

Note: `server.connected` and `server.heartbeat` are NOT wrapped in GlobalEvent — they are connection-level events emitted as bare payloads, matching OpenCode's own server behavior (global.ts lines 27-33).

#### Unchanged (`/event` endpoint for v1.4.3):

```json
{"type": "message.part.updated", "properties": {"part": {...}}}
```

### API Design

No new endpoints. Changes are limited to the SSE data format:

| Endpoint | Protocol Version | Event Format |
|----------|-----------------|--------------|
| `GET /event` | v1.4.3 and earlier | Raw event (unchanged) |
| `GET /global/event` | v1.4.4+ | GlobalEvent envelope (updated) |

---

## Security Considerations

### Threat Analysis

| Threat | Impact | Likelihood | Mitigation |
|--------|--------|------------|------------|
| Directory path leakage in SSE events | Low | High | `directory` is the working directory which is already known to the TUI client |
| Project ID exposure | Very Low | High | `project_id` is a hash of the directory path, not sensitive |
| Event injection via routing fields | Low | Very Low | Fields are server-controlled, not client-supplied |

### Security Measures

- [ ] `directory` field comes from `ServerState.working_dir` (server-controlled), not from client input
- [ ] `project` field comes from `helpers.compute_project_id()` (deterministic hash), not from client input
- [ ] No user-supplied data flows into routing fields

---

## Implementation Plan

### Phase 1: GlobalEvent Model & Serialization

- **Scope**: Add `GlobalEvent` model; create `GlobalEventFactory` on `ServerState`; update `_event_generator()` to wrap events with routing fields; exclude `server.connected`/`server.heartbeat` from wrapping
- **Deliverables**:
  - `models/events.py`: `GlobalEvent` model class
  - `state.py`: `GlobalEventFactory` class and `get_event_factory()` method on `ServerState`
  - `global_routes.py`: Updated `_event_generator()` with factory-based wrapping
- **Dependencies**: None

### Phase 2: Verification

- **Scope**: Manual testing with OpenCode v1.4.4+ TUI, automated tests
- **Deliverables**:
  - Manual test: message submission → response displayed in TUI
  - Unit tests for `GlobalEventFactory` and `GlobalEvent` model
  - Verify `/event` endpoint still works (backward compat)
  - Verify `directory` value matches `/path` endpoint output
- **Dependencies**: Phase 1

### Milestones

| Milestone | Description | Target | Status |
|-----------|-------------|--------|--------|
| GlobalEvent model | `GlobalEvent` model and factory implemented | Week 1 | Not Started |
| SSE output updated | `/global/event` produces GlobalEvent envelopes | Week 1 | Not Started |
| Manual verification | Messages display in OpenCode v1.4.4+ TUI | Week 1 | Not Started |
| Automated tests | Unit tests for GlobalEvent model and factory | Week 1 | Not Started |

### Rollback Strategy

- Revert to current `_serialize_event()` if issues arise
- The `/event` endpoint is unchanged, so v1.4.3 clients are never affected
- Feature flag not needed — the change is purely additive (adding fields to existing JSON output)

---

## Open Questions

1. **Should we fix the `/path` endpoint to also use resolved paths?**
   - Context: OpenCode's `Filesystem.resolve()` resolves symlinks using `realpathSync`. Agentpool's `/path` endpoint uses raw `working_dir` without resolution. This works because the TUI compares the GlobalEvent's `directory` against the `/path` response — as long as both use the same value, it matches. But if we later want to match OpenCode's exact behavior, both should resolve.
   - Owner: yuchen.liu
   - Status: Open (not blocking — consistency is what matters)

2. **Does the OpenCode Desktop app (`packages/app`) use the same GlobalEvent format?**
   - Context: The web app SDK (`packages/app/src/context/global-sdk.tsx`) also routes by `directory`. If we support the Desktop app in the future, the same GlobalEvent format should work.
   - Owner: yuchen.liu
   - Status: Open (investigation needed)

3. **Should wolfharness support `directory: "global"` for cross-project events?**
   - Context: OpenCode uses `directory: "global"` for events that should be delivered regardless of project context (e.g., `workspace.status`). AgentPool currently doesn't emit such events, but future features (like multi-agent status broadcasting) might need it.
   - Owner: yuchen.liu
   - Status: Open (not needed for Phase 1)

4. **Remote filesystem path handling**
   - Context: Agentpool supports remote filesystems (Docker, SSH, cloud sandboxes). When using a remote filesystem, `state.working_dir` returns a server-side path, but the TUI runs locally and expects a path that matches `project.instance.directory()`. This is a pre-existing issue with the `/path` endpoint, not introduced by this RFC.
   - Owner: yuchen.liu
   - Status: Open (pre-existing issue, out of scope)

---

## Implementation Status

> Implemented 2026-04-16. Key deviations from the original design:

| # | Deviation | Rationale |
|---|-----------|-----------|
| 1 | `GlobalEventFactory` placed in `global_routes.py` instead of `state.py` | Avoids circular imports — `state.py` cannot import from `models/events.py` without creating a dependency cycle |
| 2 | `wrap()` returns `str` (JSON) instead of `GlobalEvent` model instance | The SSE generator needs serialized JSON strings; returning the model would require the caller to serialize, adding unnecessary coupling |
| 3 | Reuses `_serialize_event()` for payload generation | The existing `_serialize_event()` already handles event serialization correctly; duplicating that logic in the factory would violate DRY |
| 4 | Uses `json.dumps(ensure_ascii=False)` instead of `model_dump_json()` | `model_dump_json()` escapes non-ASCII characters by default; `ensure_ascii=False` preserves Unicode content in event payloads |

---

## Decision Record

### Decision

**Status**: ACCEPTED

**Date**: 2026-04-16

**Approvers**: yuchen.liu

### Decision Summary

Accepted Option 2 (Formal GlobalEvent Model) with pragmatic implementation deviations documented above.

### Key Discussion Points

- Circular import issue required moving factory out of `state.py`
- Serialization strategy prioritized simplicity and DRY over strict model-driven design

### Conditions of Approval

- Must not break the existing `/event` endpoint
- Must pass manual testing with OpenCode v1.4.4+ TUI

### Dissenting Opinions

None

---

## References

### Related Documents

- [RFC-0013: Subagent Event Unification](../implemented/RFC-0013-subagent-event-unification.md)
- [RFC-0014: Spawn Session Events](../implemented/RFC-0014-spawn-session-events.md)

### External Resources

- OpenCode v1.4.4 commits: `b22add292`, `42206da1f`
- OpenCode TUI event routing: `packages/opencode/src/cli/cmd/tui/context/event.ts`
- OpenCode global SDK: `packages/opencode/src/cli/cmd/tui/context/sdk.tsx`
- OpenCode sync types: `packages/opencode/src/sync/index.ts`
- OpenCode web app SDK: `packages/app/src/context/global-sdk.tsx`

### Appendix: OpenCode v1.4.4 Protocol Changes (Detailed)

#### A.1 SSE Subscription Change

```typescript
// v1.4.3
sdk.event.subscribe({}, { signal })  // → GET /event

// v1.4.4
sdk.global.event({ signal })  // → GET /global/event
```

#### A.2 GlobalEvent Format

```typescript
// v1.4.4 GlobalEvent interface (from types.gen.ts lines 1097-1100)
interface GlobalEvent {
  directory: string;   // Required — primary routing field
  project?: string;    // Optional — informational
  workspace?: string;  // Optional — only needed when workspaces are active
  payload: Event | SyncEventPayload;
}
```

#### A.3 SyncEvent Format (NOT NEEDED for external servers)

```typescript
// v1.4.4 sync event format (for reference only)
// The TUI discards sync events in useEvent(): if (event.payload.type === "sync") return;
// Sync events are used by OpenCode's internal event sourcing system.
// Agentpool does NOT need to emit sync events for basic TUI functionality.
{ type: "sync", name: "message.updated.1", id: "evt_xxx", seq: 123, aggregateID: "sess_abc123", data: {...} }
```

#### A.4 TUI Event Routing (useEvent) — Exact Logic

```typescript
// From packages/opencode/src/cli/cmd/tui/context/event.ts
function useEvent() {
  const project = useProject();

  useEffect(() => {
    const emitter = sdk.global.event({ signal });
    emitter.on("event", (event: GlobalEvent) => {
      // 1. Global events (always delivered regardless of project)
      if (event.directory === "global") {
        handler(event.payload);
      }

      // 2. Sync events discarded (handled by separate sync system)
      if (event.payload.type === "sync") return;

      // 3. Workspace routing (priority — BLOCKS directory check)
      if (project.workspace.current()) {
        if (event.workspace === project.workspace.current()) {
          handler(event.payload);
        }
        return;  // ← directory check is SKIPPED when workspace is active
      }

      // 4. Directory routing (fallback when no workspace)
      if (event.directory === project.instance.directory()) {
        handler(event.payload);
      }
    });
  }, [project]);
}
```

#### A.5 Server Event Emission (from OpenCode's global.ts)

```typescript
// OpenCode's own server emits server.connected as a bare payload:
// (global.ts lines 27-33)
const stream = new ReadableStream({
  start(controller) {
    controller.enqueue(
      `data: ${JSON.stringify({ payload: { type: "server.connected", properties: {} } })}\n\n`
    );
    // Note: NO directory/project/workspace fields on server.connected
  }
});
```
