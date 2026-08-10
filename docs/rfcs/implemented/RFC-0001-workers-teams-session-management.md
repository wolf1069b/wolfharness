---
rfc_id: RFC-0001
title: Workers and Teams Session Management Enhancement
status: DRAFT
author: AgentPool Team
reviewers:
  - name: [TBD]
    status: pending
created: 2026-04-02
last_updated: 2026-04-02
decision_date:
related_prds: []
related_rfcs: []
---

# RFC-0001: Workers and Teams Session Management Enhancement

## Overview

This RFC proposes adding independent session management and spawn start events to Workers and Teams, matching the capabilities already present in the Subagent tool. Currently, Subagent (`task` tool) creates independent sessions with explicit `SpawnSessionStart` events, while Workers and Teams lack these features, creating inconsistency in the observability and traceability of agent execution.

The proposed changes will enable:
- Independent session tracking for Workers and Team members
- Explicit spawn/despawn lifecycle events for better observability
- Consistent event propagation across all agent delegation mechanisms
- Improved debugging and monitoring capabilities for multi-agent workflows

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

AgentPool currently supports three primary mechanisms for agent delegation:

1. **Subagent Tool** (`task`): Creates independent sessions with full lifecycle events
2. **Workers**: Runtime-registered agents/teams as tools, share parent session
3. **Teams**: Parallel or sequential execution groups, share parent session

The Subagent implementation in `subagent_tools.py` already demonstrates the desired pattern:

```python
# Generate unique session ID for the subagent run
child_session_id = identifier.ascending("session")
parent_session_id = ctx.node.session_id

# Emit SpawnSessionStart before streaming begins
spawn_event = SpawnSessionStart(
    child_session_id=child_session_id,
    parent_session_id=parent_session_id,
    tool_call_id=ctx.tool_call_id,
    spawn_mechanism="task",
    source_name=agent_or_team,
    source_type=source_type,
    depth=getattr(ctx, "current_depth", 0) + 1,
    ...
)
await ctx.events.emit_event(spawn_event)
```

However, Workers (in `workers.py`) and Teams (in `team.py`, `teamrun.py`) do not implement this pattern. They execute agents without:
- Generating independent session IDs
- Emitting `SpawnSessionStart` events
- Tracking parent-child session relationships
- Propagating depth information

### Historical Context

The Subagent tool was implemented first with full session management to support the OpenCode protocol requirements. Workers and Teams were initially designed as "lightweight" composition mechanisms without the full overhead of session tracking. As usage patterns evolved, the lack of observability in Workers and Teams has become a limitation for debugging and monitoring complex multi-agent workflows.

### Glossary

| Term | Definition |
|------|------------|
| **Session** | A unique identifier representing an execution context, tracking the lifecycle of an agent run |
| **SpawnSessionStart** | Event emitted when a new sub-session is created (child session begins) |
| **SubAgentEvent** | Wrapper event that propagates events from child agents to parent streams |
| **Worker** | An agent or team registered as a tool, callable by other agents |
| **Team** | A parallel execution group of agents/teams |
| **TeamRun** | A sequential execution chain of agents/teams |
| **Depth** | Nesting level indicating how many levels deep an agent is from the root |

---

## Problem Statement

### The Problem

Workers and Teams lack consistent session management compared to Subagent, resulting in:

1. **Inconsistent Observability**: Protocol adapters must handle different event patterns for Subagent vs Workers/Teams
2. **Broken Tracing**: Cannot trace the full execution tree when Workers or Teams are involved
3. **Missing Lifecycle Events**: No explicit spawn/despawn events for Worker/Team execution
4. **Session Overloading**: All Team members share the same session ID, making individual member tracking impossible
5. **Debugging Difficulty**: Without independent sessions, correlating logs and events to specific agent executions is challenging

### Evidence

- Protocol adapters currently detect subagent spawning by parsing tool call patterns rather than relying on explicit events
- Team member execution cannot be independently tracked in session storage
- Workers registered at runtime lack session isolation from their parent agent
- Nested Teams do not propagate session hierarchy correctly

### Impact of Inaction

- **Operational Cost**: Increased debugging time for multi-agent workflows (estimated 20-30% longer incident resolution)
- **Risk**: Inability to properly audit agent execution chains for compliance or security review
- **Opportunity Loss**: Cannot leverage session-based features (cost tracking, rate limiting, per-session configuration) for Workers and Teams

---

## Goals & Non-Goals

### Goals (In Scope)

1. **Independent Session IDs**: Workers and Team members generate unique session IDs for each execution
2. **SpawnSessionStart Events**: Emit explicit spawn events when Workers or Team members begin execution
3. **Parent-Child Tracking**: Maintain parent_session_id to child_session_id relationships
4. **Depth Propagation**: Correctly track and increment nesting depth across delegation boundaries
5. **Event Consistency**: Workers and Teams emit the same event patterns as Subagent
6. **Backward Compatibility**: Existing code continues to work without modification

### Non-Goals (Out of Scope)

1. Changing the fundamental execution model of Workers or Teams
2. Adding persistent session storage for Workers/Teams (reuse existing infrastructure)
3. Modifying Subagent behavior (already correct)
4. Adding new event types beyond SpawnSessionStart
5. Changing session ID generation algorithm

### Success Criteria

- [ ] Workers emit `SpawnSessionStart` with unique child session ID before execution
- [ ] Team members each have independent session IDs during execution
- [ ] Protocol adapters can rely on `SpawnSessionStart` events for all delegation types
- [ ] Session hierarchy correctly reflects parent-child relationships
- [ ] All existing tests pass without modification
- [ ] New tests demonstrate independent session tracking

---

## Evaluation Criteria

The following criteria will be used to objectively evaluate each option:

| Criterion | Weight | Description | Minimum Threshold |
|-----------|--------|-------------|-------------------|
| **Consistency** | High | Events and session behavior match Subagent pattern | Must match Subagent behavior |
| **Implementation Cost** | High | Development effort and code changes required | Must be completable in < 2 weeks |
| **Backward Compatibility** | High | Existing code continues to work | 100% backward compatible |
| **Performance Impact** | Medium | Overhead of session generation and event emission | < 5% latency increase |
| **Observability** | Medium | Ability to trace and monitor execution | Must enable full tracing |
| **Complexity** | Medium | Code complexity and maintenance burden | Should not significantly increase complexity |

---

## Options Analysis

### Option 1: Minimal Integration (Wrap Existing Execution)

**Description**

Add session management at the tool/execution boundary without changing internal agent execution. Generate session IDs and emit events immediately before calling existing `run()` or `run_stream()` methods.

**Advantages**

- Minimal code changes to existing Workers and Teams implementation
- Clear separation between session management and execution logic
- Easy to implement and test
- Low risk of introducing bugs in core execution paths

**Disadvantages**

- Session ID must be passed through existing method signatures
- May require changes to multiple call sites
- Event emission happens at wrapper level, not within agent itself
- Less consistent with Subagent implementation pattern

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Consistency | 4/5 | Behavior matches but implementation differs |
| Implementation Cost | 4/5 | ~1 week, minimal changes |
| Backward Compatibility | 5/5 | No breaking changes |
| Performance Impact | 4/5 | Minimal overhead |
| Observability | 4/5 | Events emitted correctly |
| Complexity | 4/5 | Adds wrapper layer |

**Effort Estimate**

- Complexity: Low
- Resources: 1 developer, 1 week
- Dependencies: None

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Incomplete coverage | Low | Medium | Comprehensive test suite |
| Event timing issues | Low | Low | Careful ordering in implementation |

---

### Option 2: Unified Base Class (Refactor Common Pattern)

**Description**

Create a shared mixin or base class that provides session management for all delegatable entities (Subagent, Workers, Teams). Refactor existing implementations to inherit from this base.

**Advantages**

- Maximum code reuse and consistency
- Single source of truth for session management logic
- Future delegation mechanisms automatically get session support
- Easier maintenance and updates

**Disadvantages**

- Requires refactoring Subagent (currently working correctly)
- Larger code change surface area
- Potential for regressions in existing functionality
- More complex testing required

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Consistency | 5/5 | Perfect consistency via shared code |
| Implementation Cost | 2/5 | ~3-4 weeks, significant refactoring |
| Backward Compatibility | 3/5 | Risk of subtle behavior changes |
| Performance Impact | 5/5 | No additional overhead |
| Observability | 5/5 | Uniform implementation |
| Complexity | 3/5 | Adds abstraction layer |

**Effort Estimate**

- Complexity: Medium-High
- Resources: 1-2 developers, 3-4 weeks
- Dependencies: Subagent refactoring

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Subagent regressions | Medium | High | Extensive test coverage |
| Breaking changes | Low | High | Careful API design |

---

### Option 3: Incremental Enhancement (Enhance Current Implementation)

**Description**

Add session management directly to Workers and Teams implementation, following the exact pattern established by Subagent. Modify `_create_agent_tool()` in workers.py and `execute()`/`run_stream()` in team.py/teamrun.py.

**Advantages**

- Follows established pattern from Subagent
- Targeted changes to specific files
- Can be implemented incrementally (Workers first, then Teams)
- Clear mapping between implementation and behavior

**Disadvantages**

- Some code duplication across Workers and Teams
- Requires understanding Subagent implementation details
- May need updates if Subagent pattern changes

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Consistency | 5/5 | Matches Subagent pattern exactly |
| Implementation Cost | 4/5 | ~1.5 weeks, moderate changes |
| Backward Compatibility | 5/5 | No breaking changes |
| Performance Impact | 4/5 | Similar to Subagent overhead |
| Observability | 5/5 | Full event support |
| Complexity | 4/5 | Straightforward additions |

**Effort Estimate**

- Complexity: Low-Medium
- Resources: 1 developer, 1.5 weeks
- Dependencies: None

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Inconsistency with Subagent | Low | Medium | Code review against Subagent |
| Missing edge cases | Low | Low | Comprehensive test suite |

---

### Options Comparison Summary

| Criterion | Option 1: Minimal | Option 2: Unified | Option 3: Incremental |
|-----------|-------------------|-------------------|----------------------|
| Consistency | 4/5 | 5/5 | 5/5 |
| Implementation Cost | 4/5 | 2/5 | 4/5 |
| Backward Compatibility | 5/5 | 3/5 | 5/5 |
| Performance Impact | 4/5 | 5/5 | 4/5 |
| Observability | 4/5 | 5/5 | 5/5 |
| Complexity | 4/5 | 3/5 | 4/5 |
| **Overall Score** | **25/30** | **23/30** | **27/30** |

---

## Recommendation

### Recommended Option

**Option 3: Incremental Enhancement**

### Justification

Option 3 provides the best balance of consistency, implementation cost, and backward compatibility. It directly follows the established Subagent pattern, ensuring behavioral consistency while minimizing risk. The incremental approach allows for:

1. **Proven Pattern**: Uses the exact implementation that already works for Subagent
2. **Manageable Scope**: Changes are localized to specific files (workers.py, team.py, teamrun.py)
3. **Incremental Delivery**: Can ship Workers support first, then Teams
4. **Low Risk**: No refactoring of working code, no breaking changes
5. **Clear Testing**: Behavior can be validated against Subagent as reference

While Option 2 (Unified Base Class) offers better long-term maintainability, the refactoring risk and higher implementation cost do not justify the benefits for this specific enhancement. Option 2 could be pursued in a future RFC focused on code organization.

### Accepted Trade-offs

1. **Code Duplication**: Some duplication between Workers and Teams implementation
   - Acceptable because the pattern is simple and stable
   - Can be addressed in future refactoring if needed

2. **No Shared Abstraction**: Each delegation mechanism implements session management separately
   - Acceptable because changes to this pattern are rare
   - Subagent pattern has been stable

### Conditions

- Implementation must include comprehensive tests matching Subagent test coverage
- Protocol adapters should be validated to work with new events
- Documentation must be updated to reflect the new capabilities

---

## Technical Design

### Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Parent Agent  │────▶│  SpawnSessionStart│────▶│  Worker/Team    │
│  (session_id=X) │     │  (child_id=Y)     │     │  (session_id=Y) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                                               │
         │                                               │
         ▼                                               ▼
┌─────────────────┐                          ┌──────────────────┐
│  SubAgentEvent  │◀─────────────────────────│  Member Events   │
│  (depth=N+1)    │                          │  (wrapped)       │
└─────────────────┘                          └──────────────────┘
```

### Key Components

#### 1. Workers Enhancement (`workers.py`)

Modify `_create_agent_tool()` to:
- Generate independent session ID via `identifier.ascending("session")`
- Emit `SpawnSessionStart` before execution
- Pass session IDs to worker's `run()` method

```python
async def worker_tool(ctx: AgentContext, prompt: str) -> str:
    # Generate session IDs
    child_session_id = identifier.ascending("session")
    parent_session_id = ctx.node.session_id or identifier.ascending("session")
    
    # Calculate depth
    current_depth = getattr(ctx, "current_depth", 0)
    
    # Emit spawn event
    spawn_event = SpawnSessionStart(
        child_session_id=child_session_id,
        parent_session_id=parent_session_id,
        tool_call_id=ctx.tool_call_id,
        spawn_mechanism="worker",
        source_name=worker_name,
        source_type=source_type,  # "agent" | "team_parallel" | "team_sequential"
        depth=current_depth + 1,
        description=f"Run worker {worker_name}",
        metadata={"prompt": prompt[:200]} if prompt else {},
    )
    await ctx.events.emit_event(spawn_event)
    
    # Execute with session context
    result = await worker.run(
        prompt,
        session_id=child_session_id,
        parent_session_id=parent_session_id,
    )
    return result
```

#### 2. Teams Enhancement (`team.py`, `teamrun.py`)

For **Parallel Teams** (`team.py`):
- Each member gets independent session ID
- Emit `SpawnSessionStart` for each member
- Wrap member events in `SubAgentEvent`

```python
async def run_stream(self, *prompts, **kwargs):
    all_nodes = list(self.nodes)
    parent_session_id = self.session_id or identifier.ascending("session")
    current_depth = getattr(kwargs, "current_depth", 0)
    
    async def wrap_stream(node, child_session_id):
        # Emit spawn event
        spawn_event = SpawnSessionStart(
            child_session_id=child_session_id,
            parent_session_id=parent_session_id,
            tool_call_id=None,
            spawn_mechanism="spawn",
            source_name=node.name,
            source_type=get_source_type(node),
            depth=current_depth + 1,
            description=f"Run team member {node.name}",
            metadata={},
        )
        await self._emit_event(spawn_event)
        
        # Stream with session context
        async for event in node.run_stream(
            *prompts,
            session_id=child_session_id,
            parent_session_id=parent_session_id,
            current_depth=current_depth + 1,
            **kwargs
        ):
            yield SubAgentEvent(
                source_name=node.name,
                source_type=get_source_type(node),
                event=event,
                depth=current_depth + 1,
                child_session_id=child_session_id,
                parent_session_id=parent_session_id,
            )
    
    # Generate session for each member
    streams = []
    for node in all_nodes:
        child_session_id = identifier.ascending("session")
        streams.append(wrap_stream(node, child_session_id))
    
    async for event in as_generated(streams):
        yield event
```

For **Sequential Teams** (`teamrun.py`):
- Similar pattern but session IDs flow through the chain
- Each step's output becomes next step's input

#### 3. BaseTeam Enhancement (`base_team.py`)

Add event emission capability:

```python
class BaseTeam(MessageNode[TDeps, TResult]):
    def __init__(self, ...):
        super().__init__(...)
        self._event_queue: asyncio.Queue | None = None
    
    async def _emit_event(self, event: RichAgentStreamEvent) -> None:
        """Emit an event through the team's event queue."""
        if self._event_queue:
            await self._event_queue.put(event)
```

#### 4. Event Type Alignment

Ensure consistent event types across all delegation mechanisms:

```python
# All mechanisms use the same spawn mechanism values
type SpawnMechanism = Literal["task", "spawn", "worker"]

# All mechanisms use the same source types
type SubAgentType = Literal[
    "agent",
    "team_parallel", 
    "team_sequential"
]
```

### Data Model Changes

No schema changes required. Existing event types (`SpawnSessionStart`, `SubAgentEvent`) already support the needed fields. The enhancement is in event emission, not event structure.

### API Changes

No public API changes. The modifications are internal implementation details that do not affect:
- YAML configuration format
- Public Python API
- Protocol interfaces (ACP, AG-UI, OpenCode)

---

## Security Considerations

### Threat Analysis

| Threat | Impact | Likelihood | Mitigation |
|--------|--------|------------|------------|
| Session ID collision | Medium | Very Low | Uses cryptographically secure generation |
| Event injection | Low | Low | Events are internal, not user-controlled |
| Information leakage via metadata | Low | Low | Metadata only includes prompt preview (truncated) |

### Security Measures

- [x] Session IDs generated using `identifier.ascending()` with secure random component
- [x] No sensitive data in event metadata
- [x] Session relationships are read-only after creation

---

## Implementation Plan

### Phases

#### Phase 1: Workers Session Support

- **Scope**: Add session management to WorkersTools in `workers.py`
- **Deliverables**:
  - `_create_agent_tool()` generates session IDs
  - `SpawnSessionStart` event emission
  - Session ID propagation to worker execution
  - Unit tests for Worker session management
- **Dependencies**: None

#### Phase 2: Teams Session Support

- **Scope**: Add session management to Team and TeamRun
- **Deliverables**:
  - `BaseTeam._emit_event()` method
  - `Team.run_stream()` session support
  - `TeamRun.run_stream()` session support
  - Unit tests for Team session management
- **Dependencies**: Phase 1 (for pattern validation)

#### Phase 3: Integration & Validation

- **Scope**: End-to-end testing and protocol adapter validation
- **Deliverables**:
  - Integration tests with nested Workers and Teams
  - Protocol adapter validation (ACP, AG-UI, OpenCode)
  - Documentation updates
  - Performance benchmarks
- **Dependencies**: Phase 1, Phase 2

### Milestones

| Milestone | Description | Target | Status |
|-----------|-------------|--------|--------|
| Workers Implementation | Workers emit SpawnSessionStart with independent sessions | Week 1 | Not Started |
| Teams Implementation | Teams emit SpawnSessionStart with member sessions | Week 2 | Not Started |
| Integration Testing | E2E tests and protocol validation | Week 2.5 | Not Started |
| Documentation | API docs and migration guide | Week 3 | Not Started |

### Rollback Strategy

If issues are discovered:

1. **Workers**: Can be rolled back by reverting `workers.py` changes
2. **Teams**: Can be rolled back by reverting `team.py` and `teamrun.py` changes
3. **No data migration required** - session IDs are ephemeral
4. **Feature flags**: Consider adding `enable_worker_sessions` config flag for gradual rollout

---

## Open Questions

1. **Should Team members share session state?**
   - Context: Currently Team members are independent; should they share any session-scoped variables?
   - Owner: Architecture team
   - Status: Open

2. **How should Workers handle `pass_message_history` with independent sessions?**
   - Context: Workers have `pass_message_history` option; should this work across session boundaries?
   - Owner: AgentPool maintainers
   - Status: Open

3. **Should we add `SpawnSessionEnd` events for symmetry?**
   - Context: Currently only have start events; would end events be useful?
   - Owner: Protocol team
   - Status: Open

4. **What is the performance impact on high-frequency Worker calls?**
   - Context: Session ID generation and event emission add overhead
   - Owner: Performance team
   - Status: Open

---

## Decision Record

> To be completed after RFC review

### Decision

**Status**: [PENDING REVIEW]

**Date**: 

**Approvers**:
- 

### Decision Summary

[To be filled]

### Key Discussion Points

1. 

### Conditions of Approval

-

### Dissenting Opinions

-

---

## References

### Related Documents

- Subagent implementation: `src/wolfharness_toolsets/builtin/subagent_tools.py`
- Workers implementation: `src/wolfharness_toolsets/builtin/workers.py`
- Team implementation: `src/wolfharness/delegation/team.py`
- TeamRun implementation: `src/wolfharness/delegation/teamrun.py`
- Event definitions: `src/wolfharness/agents/events/events.py`
- Session ID generation: `src/wolfharness/utils/identifiers.py`

### External Resources

- Agent Communication Protocol (ACP) specification
- AG-UI protocol documentation

### Appendix

#### A. Current vs Proposed Event Flow

**Current (Workers)**:
```
Parent Agent ──▶ Worker.run() ──▶ Result
   (session=X)     (session=X)
```

**Proposed (Workers)**:
```
Parent Agent ──▶ SpawnSessionStart ──▶ Worker.run() ──▶ Result
   (session=X)   (child_session=Y)      (session=Y)
```

**Current (Teams)**:
```
Team.run() ──▶ Member1.run()  (session=X)
          ──▶ Member2.run()  (session=X)
```

**Proposed (Teams)**:
```
Team.run() ──▶ SpawnSessionStart ──▶ Member1.run()  (session=Y1)
          ──▶ SpawnSessionStart ──▶ Member2.run()  (session=Y2)
```

#### B. Test Plan

1. **Unit Tests**:
   - Workers emit `SpawnSessionStart` with correct fields
   - Team members each get unique session ID
   - Parent-child session relationships correct
   - Depth increments correctly

2. **Integration Tests**:
   - Nested Workers and Teams
   - Mixed delegation (Subagent calling Worker calling Team)
   - Protocol adapter event handling

3. **Performance Tests**:
   - Session ID generation overhead
   - Event emission latency
   - High-frequency Worker call benchmarks
