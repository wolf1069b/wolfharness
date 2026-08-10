---
rfc_id: RFC-0032
title: "ACP Slash Commands Protocol Compliance: Move from initialize to session/update"
status: DRAFT
author: yuchen.liu
reviewers:
  - name: Oracle
    status: completed
  - name: Metis
    status: completed
created: 2026-05-26
last_updated: 2026-05-26
decision_date:
related_rfcs:
  - RFC-0016 (Skill Slash Commands)
  - RFC-0031 (ACP Server Per-Session Agent Isolation)
---

# RFC-0032: ACP Slash Commands Protocol Compliance

## Overview

This RFC proposes aligning AgentPool's ACP slash command advertisement with the official Agent Client Protocol (ACP) specification. Currently, AgentPool declares available slash commands during the `initialize` handshake via `AgentCapabilities.slash_commands`. The ACP specification mandates that slash commands be advertised **after** session creation through the `session/update` notification with `available_commands_update`. This RFC outlines the migration path to remove `slash_commands` from `AgentCapabilities` and rely exclusively on the per-session `session/update` mechanism — which AgentPool already partially implements.

## Table of Contents

- Background & Context
- [Problem Statement](#problem-statement)
- Goals & Non-Goals
- [Evaluation Criteria](#evaluation-criteria)
- [Options Analysis](#options-analysis)
- [Recommendation](#recommendation)
- [Technical Design](#technical-design)
- [Review Findings](#review-findings)
- Implementation Plan
- [Open Questions](#open-questions)
- Decision Record
- [References](#references)

---

## Background & Context

### Current State

AgentPool's ACP server (`AgentPoolACPAgent`) advertises slash commands in two places:

1. **`initialize` response** (`acp_agent.py:482-505`): The `initialize()` method builds `InitializeResponse` with `slash_commands=skill_commands`, populating `AgentCapabilities.slash_commands` in the JSON-RPC response.
2. **`session/update` notification** (`session.py:562-572`): The `ACPSession.send_available_commands_update()` method sends `AvailableCommandsUpdate` via `ACPNotifications.update_commands()` after session creation.

The schema layer supports both paths:
- `AgentCapabilities.slash_commands: list[AvailableCommand]` (`capabilities.py:273`)
- `InitializeResponse.create(slash_commands=...)` (`agent_responses.py:284-339`)
- `AvailableCommandsUpdate` (`session_updates.py:354-363`)

**Current `initialize` response path** (`acp_agent.py:490-505`):
```python
skill_commands = self.get_skill_commands()
return InitializeResponse.create(
    protocol_version=version,
    name="wolfharness",
    title="AgentPool",
    version=_version("wolfharness"),
    # ... other capabilities ...
    slash_commands=skill_commands,  # ← ADVERTISED AT INIT TIME
)
```

**Current `session/update` path** (`acp_agent.py:547-550`, `621`, `732`):
```python
# After new_session / load_session / resume_session
self.tasks.create_task(session.send_available_commands_update())
```

The `session/update` path is already invoked after `new_session()`, `load_session()`, and `resume_session()`, meaning AgentPool currently sends slash commands **twice**: once globally at initialization, and once per session.

### ACP Protocol Specification

The official ACP specification (`agent-client-protocol/docs/protocol/slash-commands.mdx`) states:

> After creating a session, the Agent **MAY** send a list of available commands via the `available_commands_update` session notification.

The spec provides this example:
```json
{
  "jsonrpc": "2.0",
  "method": "session/update",
  "params": {
    "sessionId": "sess_abc123def456",
    "update": {
      "sessionUpdate": "available_commands_update",
      "availableCommands": [...]
    }
  }
}
```

Key protocol requirements:
- Commands are **per-session**, not global
- Commands can be **dynamically updated** during a session
- Commands are advertised via `session/update`, not `initialize`

### Glossary

| Term | Definition |
|------|------------|
| `AgentCapabilities` | ACP schema object advertised in `initialize` response |
| `AvailableCommandsUpdate` | ACP session update type for advertising slash commands |
| `session/update` | ACP JSON-RPC notification method for session state changes |
| `initialize` | ACP JSON-RPC method for capability negotiation |
| `ACPSkillBridge` | AgentPool component that exposes skill commands as slash commands |

---

## Problem Statement

### The Problem

AgentPool's current implementation violates the ACP protocol specification for slash command advertisement:

1. **Wrong lifecycle phase**: Commands are advertised during `initialize` (global/static) rather than after `session/new` (per-session/dynamic).
2. **Double advertisement**: Commands are sent both at initialization and per-session, causing redundant protocol traffic.
3. **Schema drift**: `AgentCapabilities.slash_commands` is not part of the official ACP spec for the `initialize` response. While the field exists in AgentPool's schema, it has no equivalent in the protocol's `AgentCapabilities` definition.
4. **Per-session semantics lost**: By advertising at initialization, AgentPool implies commands are global and static. In reality, commands vary per session based on: agent type, loaded skills, MCP server prompts, prompt hub commands, and session-specific context.

### Evidence

- `capabilities.py:273`: `slash_commands: list[AvailableCommand]` defined on `AgentCapabilities`
- `agent_responses.py:301`: `InitializeResponse.create()` accepts `slash_commands` parameter
- `acp_agent.py:504`: `slash_commands=skill_commands` passed to `InitializeResponse.create()`
- `slash-commands.mdx:10`: "After creating a session, the Agent MAY send a list of available commands via the `available_commands_update` session notification"
- `slash-commands.mdx:71-73`: "The Agent can update the list of available commands at any time during a session by sending another `available_commands_update` notification"

### Impact of Inaction

- **Risk**: ACP-compliant clients may ignore `initialize`-time `slash_commands` entirely, causing skill commands to be invisible until a session is created — but since some clients rely on the spec-compliant `session/update` path, they will work.
- **Risk**: Non-compliant clients that only look at `initialize`-time `slash_commands` will break when AgentPool eventually aligns with the spec.
- **Risk**: Protocol divergence makes AgentPool harder to integrate with third-party ACP clients that strictly follow the specification.
- **Cost**: Maintaining the dual-path approach increases schema and test complexity.

---

## Goals & Non-Goals

### Goals (In Scope)

1. Remove `slash_commands` from `AgentCapabilities` schema and `InitializeResponse`
2. Ensure all slash command advertisement flows through `session/update` (`AvailableCommandsUpdate`)
3. Maintain backward compatibility for existing ACP clients during a deprecation window
4. Update all tests that assert on `initialize`-time `slash_commands`
5. Update documentation to reflect the protocol-compliant behavior

### Non-Goals (Out of Scope)

1. **Not**: Changing the content or semantics of available commands themselves
2. **Not**: Adding new command types or command registration mechanisms
3. **Not**: Refactoring `ACPSkillBridge` or skill command discovery logic
4. **Not**: Modifying how commands are executed (`process_prompt` / `execute_slash_command`)
5. **Not**: Changing the ACP protocol spec — this RFC aligns AgentPool with the existing spec

### Success Criteria

- [ ] `initialize` response no longer contains `slash_commands` in `agent_capabilities`
- [ ] `AgentCapabilities` schema no longer has a `slash_commands` field
- [ ] All existing tests pass after updating assertions
- [ ] `session/send_available_commands_update()` continues to work after session creation
- [ ] ACP clients receive commands via `session/update` as per spec
- [ ] No regression in skill command visibility for supported clients

---

## Evaluation Criteria

| Criterion | Weight | Description | Minimum Threshold |
|-----------|--------|-------------|-------------------|
| Protocol Compliance | Critical | Aligns with ACP spec `slash-commands.mdx` | Must remove `slash_commands` from `initialize` |
| Backward Compatibility | High | Existing clients continue to work | No client-visible regressions for 1 release cycle |
| Minimality | High | Smallest change that achieves compliance | ≤ 6 files modified |
| Test Coverage | High | All affected tests updated | 100% of `slash_commands`-related tests updated |
| Documentation | Medium | Docs reflect new behavior | At least 1 doc page updated |

---

## Options Analysis

### Option 1: Complete Removal from `initialize` (Recommended)

Remove `slash_commands` entirely from `AgentCapabilities`, `InitializeResponse.create()`, and `AgentPoolACPAgent.initialize()`. Rely solely on the existing `session/update` (`AvailableCommandsUpdate`) path that is already invoked after `new_session`, `load_session`, and `resume_session`.

**Advantages**:
- Full protocol compliance — aligns exactly with `slash-commands.mdx`
- Eliminates double advertisement (reduced protocol traffic)
- Correct per-session semantics — commands are scoped to the session
- Minimal code change — the `session/update` path already exists and works
- Removes schema drift between AgentPool and the ACP specification

**Disadvantages**:
- Clients that relied solely on `initialize`-time `slash_commands` will no longer see commands until session creation
- Requires updating test assertions in `test_capabilities.py` and integration tests
- May require a brief deprecation window if external consumers depend on the field

**Evaluation Against Criteria**:

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Protocol Compliance | ✅ Excellent | Fully aligns with ACP spec |
| Backward Compatibility | ⚠️ Moderate | Clients must support `session/update`; most already do |
| Minimality | ✅ Excellent | ~5 files, removes code rather than adding |
| Test Coverage | ✅ Good | Update existing tests, no new test infrastructure needed |
| Documentation | ✅ Good | Update existing docs to remove `initialize` references |

**Effort Estimate**:
- Complexity: Low
- Resources: 1 engineer, 0.5–1 day
- Dependencies: None

**Risk Assessment**:

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Client incompatibility | Low | Medium | Major clients (Zed, Toad) already handle `session/update`; verify during testing |
| Test regressions | Low | Low | Tests are mechanical assertion updates |
| External consumer breakage | Low | Medium | `AgentCapabilities` is internal to AgentPool; external clients parse JSON |

---

### Option 2: Deprecation with Fallback

Keep `AgentCapabilities.slash_commands` but set it to an empty list in `initialize()`. Continue sending actual commands via `session/update`. Add a deprecation comment/note indicating the field will be removed in a future release.

**Advantages**:
- Maximum backward compatibility — field still exists in schema
- Zero risk of breaking external consumers that depend on the field
- Gradual migration path for any non-compliant clients

**Disadvantages**:
- Not protocol compliant — the field should not exist in `AgentCapabilities` at all
- Perpetuates schema drift from the ACP specification
- Increases technical debt — the field becomes dead code with no spec backing
- Confusing for new developers — "why does this field exist if it's always empty?"

**Evaluation Against Criteria**:

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Protocol Compliance | ❌ Poor | Field still exists, contradicting spec |
| Backward Compatibility | ✅ Excellent | Zero breaking changes |
| Minimality | ⚠️ Moderate | Must keep and maintain dead code |
| Test Coverage | ✅ Good | Existing tests pass with minor modifications |
| Documentation | ❌ Poor | Must document a deprecated, non-spec field |

**Effort Estimate**:
- Complexity: Low
- Resources: 1 engineer, 0.5 day
- Dependencies: None

**Risk Assessment**:

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Technical debt accumulation | High | Medium | Schedule removal in next release |
| Developer confusion | Medium | Low | Add prominent deprecation comments |

---

### Option 3: Keep Both Paths with Client Detection

Retain `initialize`-time `slash_commands` and add client capability detection: only send `slash_commands` in `initialize` if the client advertises that it does not support `session/update` notifications. Otherwise, rely on `session/update`.

**Advantages**:
- Backward compatible for all clients
- Spec-compliant for modern clients
- Graceful degradation for legacy clients

**Disadvantages**:
- Over-engineered — no known client lacks `session/update` support
- Adds complexity to `initialize()` logic
- No ACP capability flag exists for "does not support session/update" — would require inventing one
- Perpetuates the schema drift problem

**Evaluation Against Criteria**:

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Protocol Compliance | ⚠️ Moderate | Still sends non-spec field conditionally |
| Backward Compatibility | ✅ Excellent | Supports all clients |
| Minimality | ❌ Poor | Adds conditional logic and invented capability flags |
| Test Coverage | ❌ Poor | Requires testing both code paths |
| Documentation | ❌ Poor | Must document invented capability protocol |

**Effort Estimate**:
- Complexity: Medium
- Resources: 1 engineer, 2–3 days
- Dependencies: None

---

### Options Comparison Summary

| Criterion | Option 1: Complete Removal | Option 2: Deprecation | Option 3: Client Detection |
|-----------|---------------------------|----------------------|---------------------------|
| Protocol Compliance | ✅ Full | ❌ Partial | ⚠️ Conditional |
| Backward Compatibility | ⚠️ One release window | ✅ Perfect | ✅ Perfect |
| Minimality | ✅ ~5 files | ⚠️ Dead code | ❌ Over-engineered |
| Test Coverage | ✅ Straightforward | ✅ Straightforward | ❌ Complex |
| Maintenance Burden | ✅ Low | ⚠️ Medium | ❌ High |
| **Overall** | **Recommended** | Rejected | Rejected |

---

## Recommendation

**Option 1: Complete Removal from `initialize`.**

The `session/update` path for slash commands is already fully implemented and tested. Removing the `initialize`-time path is a net reduction in code and aligns AgentPool with the ACP specification. The risk of client breakage is low because:

1. The `session/update` notification is a baseline ACP requirement — all compliant clients must support it
2. AgentPool already sends commands via `session/update` after every session creation
3. No known ACP client relies solely on `initialize`-time `slash_commands` for command discovery

### Accepted Trade-offs

1. **Brief deprecation window**: The `slash_commands` field will be removed in the next minor release. No formal deprecation cycle is needed because the field is not part of the public ACP spec.
2. **Client adaptation**: Any client that only reads commands from `initialize` will need to adapt. This is considered acceptable because such a client would already be non-compliant with the ACP specification.

### Conditions

- All tests must pass after the change
- `session/send_available_commands_update()` must be verified to work correctly in integration tests
- Documentation must be updated before marking RFC as COMPLETED

---

## Technical Design

### Architecture Overview

```
BEFORE (Current — Non-Compliant):
┌─────────────────┐     ┌─────────────────────────────┐
│  Client         │────▶│  initialize                 │
│                 │     │  └─ AgentCapabilities       │
│                 │◀────│     └─ slash_commands[]    │  ← WRONG PHASE
└─────────────────┘     └─────────────────────────────┘
         │
         │ session/new
         ▼
┌─────────────────────────────┐
│  session/update             │
│  └─ available_commands_update│  ← CORRECT (already done)
│     └─ availableCommands[]  │
└─────────────────────────────┘

AFTER (Protocol-Compliant):
┌─────────────────┐     ┌─────────────────────────────┐
│  Client         │────▶│  initialize                 │
│                 │     │  └─ AgentCapabilities       │
│                 │◀────│     (no slash_commands)    │  ← REMOVED
└─────────────────┘     └─────────────────────────────┘
         │
         │ session/new
         ▼
┌─────────────────────────────┐
│  session/update             │
│  └─ available_commands_update│  ← SOLE PATH
│     └─ availableCommands[]  │
└─────────────────────────────┘
```

### Key Components

#### 1. `AgentCapabilities` Schema Change

**File**: `src/acp/schema/capabilities.py`

**Remove**:
```python
slash_commands: list[AvailableCommand] = Field(default_factory=list)
```

**Update `AgentCapabilities.create()`**: Remove `slash_commands` parameter and its usage in the method body.

#### 2. `InitializeResponse` Schema Change

**File**: `src/acp/schema/agent_responses.py`

**Update `InitializeResponse.create()`**: Remove `slash_commands` parameter and its forwarding to `AgentCapabilities.create()`.

#### 3. `AgentPoolACPAgent.initialize()` Update

**File**: `src/wolfharness_server/acp_server/acp_agent.py`

**Remove**:
```python
skill_commands = self.get_skill_commands()
# ...
slash_commands=skill_commands,
```

The `get_skill_commands()` method on `AgentPoolACPAgent` is currently used **only** by `initialize()`. After removal, it becomes dead code and should be evaluated for deletion. Note: `ACPSession.send_available_commands_update()` does **not** call `get_skill_commands()` — it calls `self.get_acp_commands()` which operates on the session's `command_store` directly (`session.py:614-628`).

#### 4. Test Updates

**File**: `tests/acp/schema/test_capabilities.py`

- Remove or repurpose all `TestAgentCapabilitiesSlashCommands` test methods
- Add tests verifying that `AgentCapabilities` does **not** contain `slash_commands` after deserialization from old JSON (backward compat)

**File**: `tests/servers/acp_server/test_acp_skill_commands.py`

- **Critical**: Tests in this file (e.g., `test_initialize_exposes_skill_commands`, `test_initialize_without_skills_has_empty_commands`) are **fundamentally testing removed behavior**. These must be **rewritten or deleted**, not merely updated with new assertions.
- Replace with tests verifying that `initialize()` returns `AgentCapabilities` without `slash_commands`, and that commands are received via `session/update` notification after session creation.

**Files**: `tests/server/acp/test_skill_commands.py`, `tests/integration/test_skill_commands_e2e.py`

- Update any tests that assert on `initialize`-time `slash_commands`
- Add assertions verifying commands arrive via `session/update` instead
- Verify e2e/integration tests do not have hidden assertions on the removed field

#### 5. Documentation Updates

**File**: `docs/features/skill-commands.md`

- Remove any references to `initialize`-time command advertisement
- Clarify that commands are advertised per-session via `session/update`

---

## Implementation Plan

### Phase 1: Schema and Agent Layer Changes

**Scope**: Remove `slash_commands` from schema and `initialize()`

**Files**:
| File | Changes |
|------|---------|
| `src/acp/schema/capabilities.py` | Remove `slash_commands` field from `AgentCapabilities`; update `create()` |
| `src/acp/schema/agent_responses.py` | Remove `slash_commands` parameter from `InitializeResponse.create()` |
| `src/wolfharness_server/acp_server/acp_agent.py` | Remove `slash_commands=skill_commands` from `initialize()` |

**Duration**: 0.5 day

### Phase 2: Test Updates

**Scope**: Rewrite/delete tests that assert on `initialize`-time `slash_commands`; add tests for `session/update` path

**Files**:
| File | Changes |
|------|---------|
| `tests/acp/schema/test_capabilities.py` | Remove `TestAgentCapabilitiesSlashCommands`; add backward-compat deserialization test |
| `tests/servers/acp_server/test_acp_skill_commands.py` | **Rewrite**: replace `initialize`-time tests with `session/update` path tests |
| `tests/server/acp/test_skill_commands.py` | Update assertions |
| `tests/integration/test_skill_commands_e2e.py` | Update assertions; verify `session/update` path end-to-end |

**Duration**: 0.5–1 day

### Phase 3: Documentation and Validation

**Scope**: Update docs and run full test suite

**Files**:
| File | Changes |
|------|---------|
| `docs/features/skill-commands.md` | Remove `initialize` references; clarify `session/update` path |

**Validation**:
- `pytest tests/acp/schema/`
- `pytest tests/server/acp/`
- `pytest tests/integration/test_skill_commands_e2e.py`
- `pytest tests/servers/acp_server/test_acp_skill_commands.py`

**Duration**: 0.5 day

### Rollback Strategy

Revert by restoring:
1. `slash_commands` field in `AgentCapabilities`
2. `slash_commands` parameter in `InitializeResponse.create()`
3. `slash_commands=skill_commands` in `AgentPoolACPAgent.initialize()`
4. Original test assertions

---

## Review Findings

### Metis Review (2026-05-26)

**Ambiguities and AI Failure Points Identified**:

1. **Backward compatibility contradiction** (Addressed): The RFC originally claimed backward compatibility as a goal while rejecting the only backward-compatible option (deprecation). The "Accepted Trade-offs" section has been updated to clarify that no formal deprecation cycle is needed because the field is not part of the public ACP spec, but implementers should verify no external consumers depend on it.

2. **`get_skill_commands()` usage analysis error** (Addressed): The original RFC incorrectly stated that `get_skill_commands()` is used by `send_available_commands_update()`. Code inspection shows `send_available_commands_update()` calls `self.get_acp_commands()` (session-level) instead. `get_skill_commands()` is only used by `initialize()` and becomes dead code after removal. The Technical Design section has been corrected.

3. **Test scope underestimated** (Addressed): The original RFC described test changes as "assertion updates." In reality, `tests/servers/acp_server/test_acp_skill_commands.py` contains tests whose entire premise is `initialize`-time command exposure — these must be rewritten or deleted, not patched. The Implementation Plan now explicitly calls out test rewriting.

4. **Race condition: `session/update` timing** (Documented): `send_available_commands_update()` is scheduled as a background task (`self.tasks.create_task()`) after the `session/new` response is returned. There is no ordering guarantee between the response and the notification. A fast client could query for commands before the async task runs. The current behavior is accepted as-is because:
   - The gap is typically one network round-trip
   - All compliant ACP clients must support `session/update`
   - Commands are per-session by design — there is no valid use case for commands before session creation

5. **Missing edge cases** (Added to Open Questions):
   - Session creation failure after successful `initialize`: client gets zero commands (acceptable per spec)
   - Empty command lists: `send_available_commands_update()` sends `availableCommands: []` — this is spec-compliant
   - Mid-session command updates: Already supported via `_register_mcp_prompts_as_commands()` and `_register_prompt_hub_commands()`

### Oracle Review (2026-05-26)

**Technical Assessment**:

1. **Recommended approach is correct** (Confirmed): The `session/update` path is already fully implemented and robust. Removing the `initialize`-time path is a net code reduction with zero new infrastructure needed.

2. **Schema safety verified** (Confirmed): `AgentCapabilities` inherits from `AnnotatedObject` → Pydantic `BaseModel`. Pydantic v2 default is `extra='ignore'`, so old JSON with `slash_commands` will deserialize safely. However, implementers should add an explicit backward-compat test.

3. **Client impact: Low** (Confirmed): `src/acp/client/` has zero references to `slash_commands`. Major ACP clients (Zed, Toad) strictly follow the spec and already handle `session/update`. The risk of breaking real clients is low.

4. **Missing: Deprecation warning phase** (Recommendation): Oracle recommends a hybrid approach:
   - **Phase 1** (this release): Set `slash_commands=[]` in `initialize()`, keep field in schema, emit `DeprecationWarning`
   - **Phase 2** (next minor release): Remove field entirely
   This costs ~1 line (`warnings.warn(...)`) and provides measurable safety. The RFC author has considered this and decided on hard removal due to the field being non-spec, but acknowledges the risk.

5. **`InitializeResponse.create()` docstring bug** (Drive-by): The docstring incorrectly says "Create an instance of AgentCapabilities" — it creates an `InitializeResponse`. This pre-existing bug should be fixed as a drive-by.

6. **Dynamic command updates: Verified** (Resolved): `session.py:582` calls `send_available_commands_update()` after `_register_mcp_prompts_as_commands()`. `session.py:268-272` handles nested ACP agent command updates. This is already complete.

7. **Criteria weighting: Appropriate** (Confirmed): Protocol Compliance (Critical), Backward Compatibility (High), Minimality (High), Test Coverage (High), Documentation (Medium) are correctly weighted.

### Oracle + Metis Consensus

- **Core recommendation (Option 1) is sound** and should proceed
- **Test rewriting is required**, not just assertion updates
- **`get_skill_commands()` should be evaluated for deletion** after removal
- **Add backward-compat deserialization test** as a blocking condition
- **Verify `extra='ignore'` on `AnnotatedObject`** before merge
- **Close Open Question #3** — dynamic updates are already handled

---

## Open Questions

1. **External client dependency on `initialize`-time `slash_commands`**
   - Context: Are there any external ACP clients (outside AgentPool's test suite) that read commands from `initialize`?
   - Owner: yuchen.liu
   - Status: Open — investigate before merging

2. **`AgentCapabilities` backward compatibility**
   - Context: Old JSON with `slash_commands` key must still deserialize without errors after removing the field. `AnnotatedObject` inherits from Pydantic `BaseModel`; Pydantic v2 default is `extra='ignore'`, so unknown fields are dropped safely. This should still be verified with an explicit test.
   - Owner: Implementer
   - Status: **RESOLVED in design** — add backward-compat deserialization test as blocking condition; verify `extra='ignore'` on `AnnotatedObject`

3. **Dynamic command updates during session**
   - Context: The ACP spec allows commands to be updated at any time. Code inspection confirms AgentPool already supports this: `session.py:582` calls `send_available_commands_update()` after `_register_mcp_prompts_as_commands()`, and `session.py:268-272` handles nested ACP agent command updates.
   - Owner: Implementer
   - Status: **RESOLVED** — verified existing implementation covers dynamic updates

4. **Client verification**
   - Context: Has any production ACP client been verified to work without `initialize`-time `slash_commands`? The ACP baseline spec (`capabilities.py:207-208`) states all agents MUST support `session/update`, so compliant clients are expected to handle it.
   - Owner: yuchen.liu
   - Status: Open — verify with Zed/Toad before merge

5. **`get_skill_commands()` dead code**
   - Context: After removing `slash_commands` from `initialize()`, `AgentPoolACPAgent.get_skill_commands()` has no remaining callers. It should be evaluated for deletion or retained if future features need it.
   - Owner: Implementer
   - Status: Open — decide during implementation

---

## Decision Record

> To be completed after RFC review.

### Decision

**Status**: PENDING REVIEW

**Date**:

**Approvers**:
- [Name 1]
- [Name 2]

### Decision Summary

[To be filled after review]

### Key Discussion Points

1. [Point 1]
2. [Point 2]

### Conditions of Approval

[To be filled after review]

---

## References

### Related Documents

- [ACP Slash Commands Protocol Spec](../../agent-client-protocol/docs/protocol/slash-commands.mdx)
- [RFC-0016: Skill Slash Commands](./RFC-0016-skill-slash-commands.md)
- [RFC-0031: ACP Server Per-Session Agent Isolation](../implemented/RFC-0031-acp-per-session-agent-isolation.md)

### Code References

- `src/acp/schema/capabilities.py:273` — `AgentCapabilities.slash_commands`
- `src/acp/schema/agent_responses.py:284-339` — `InitializeResponse.create()`
- `src/wolfharness_server/acp_server/acp_agent.py:482-505` — `AgentPoolACPAgent.initialize()`
- `src/wolfharness_server/acp_server/session.py:562-572` — `ACPSession.send_available_commands_update()`
- `src/acp/agent/notifications.py:336-339` — `ACPNotifications.update_commands()`
- `src/acp/schema/session_updates.py:354-363` — `AvailableCommandsUpdate`

### External Resources

- [Agent Client Protocol — Slash Commands](https://agentclientprotocol.com/protocol/slash-commands)
