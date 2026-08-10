---
rfc_id: RFC-0014
title: SpawnSessionStart Event for Explicit Subsession Creation
status: DRAFT
author: AgentPool Team
reviewers: []
created: 2026-02-14
last_updated: 2026-02-14
---

# RFC-0014: Adding SpawnSessionStart Event for Explicit Subsession Creation

## Overview

This RFC proposes adding a single new event type - `SpawnSessionStart` - to the AgentPool event system. This event provides an explicit signal when a subsession (spawn/subagent) is created, eliminating the need for protocol adapters (ACP, OpenCode, AG-UI) to hardcode detection of specific tool calls.

**Note**: After discussion, we determined that a close event is unnecessary because `StreamCompleteEvent` already implicitly signals subsession completion (see "Why No Close Event?" section).

### Problem Statement

The current event system relies on `SubAgentEvent` wrappers to propagate subagent activity, but lacks an explicit lifecycle signal for when a **new** subsession is created. Protocol adapters currently work around this by:

1. Hardcoding checks for tool names like `"task"`, `"spawn"`, or `"subagent"`
2. Inferring session boundaries from `RunStartedEvent` with `parent_session_id`
3. Creating child sessions reactively on the first `SubAgentEvent` occurrence

```python
# Current workaround in event_processor.py (lines 652-702)
if child_session_id and child_ctx is None:
    # Creating session because this is the first SubAgentEvent
    # This is implicit, not explicit
    await ctx.state.ensure_session(child_session_id, parent_id=ctx.session_id)
```

This approach is brittle and requires protocol adapters to have internal knowledge of tool implementation details.

## Proposed Solution

### Single Event: SpawnSessionStart

**Rationale**: Only a start event is needed because:

- **`StreamCompleteEvent` already handles completion**: When a subagent finishes, it emits `StreamCompleteEvent` wrapped in `SubAgentEvent.child_session_id`. Protocol adapters can detect this as the close signal.
- **Simpler design**: Fewer event types reduce complexity and cognitive load.
- **Backward compatible**: Adding one event is less disruptive than adding two.

### Schema

```python
@dataclass(kw_only=True)
class SpawnSessionStart:
    """Signals the creation of a new subsession (spawn).
    
    Emitted BEFORE any content events from the spawned session.
    Protocol adapters should use this to initialize child session state
    and create container UI elements.
    
    The subsession lifecycle is:
    1. SpawnSessionStart -> Create child session
    2. SubAgentEvent (with events like PartDeltaEvent, ToolCallStartEvent...) -> Route to child
    3. SubAgentEvent.child_event=StreamCompleteEvent -> Finalize child session
    """
    
    child_session_id: str
    """The unique ID of the newly created child session."""
    
    parent_session_id: str
    """The ID of the parent session that spawned the child."""
    
    tool_call_id: str | None
    """The tool call ID that triggered the spawn (if applicable)."""
    
    spawn_mechanism: Literal["sync", "async_worker", "manual"]
    """The mechanism that created this spawn:
    - "sync": Synchronous task tool execution (blocks until completion)
    - "async_worker": Background/async worker execution (non-blocking)
    - "manual": Manually triggered via code (e.g., internal delegation)
    """
    
    source_name: str
    """Name of the agent/team that will execute in the child session."""
    
    source_type: Literal["agent", "team_parallel", "team_sequential"]
    """Type of node executing in the child session."""
    
    depth: int = 1
    """Nesting depth of this spawn (1 = direct child, 2 = grandchild, etc.).
    Used for depth limitation and UI rendering hierarchy."""
    
    description: str
    """Human-readable description of what the spawned session will do."""
    
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata for the spawn. May include:
    - task_id: For async tasks
    - prompt: Summary of instructions (truncated)
    - user_message_id: ID of the prompting user message in parent session
    - Other tool-specific metadata
    """
    
    event_kind: Literal["spawn_session_start"] = "spawn_session_start"
    """Event type identifier for dispatch."""
```

### Complete Event Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    New Event Flow with SpawnSessionStart                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  task() starts                                                           │
│     │                                                                    │
│     ▼                                                                    │
│  SpawnSessionStart (NEW)  <- Explicit signal to create child session    │
│     │                      ┌─────────────────────────────────────┐       │
│     ▼                      │ Protocol adapters can now:          │       │
│                             │ - Create child session upfront      │       │
│  SubAgentEvent              │ - Create container UI element       │       │
│  ├- PartStartEvent          │ - Attach spawn metadata             │       │
│  ├- PartDeltaEvent          └─────────────────────────────────────┘       │
│  ├- ToolCallStartEvent                                                   │
│  ├- ToolCallProgressEvent                                                │
│  ├- ...                                                                  │
│  └- StreamCompleteEvent <- Implicit close signal                         │
│                            (existing behavior, no change needed)         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Why No Close Event?

After analysis, we determined `SpawnSessionClose` is unnecessary:

### Existing Close Signal Already Works

```python
# event_processor.py already handles this (lines 669-782)
async def _process_subagent_event(self, subagent_event, ctx):
    # ... process child events ...
    
    # Line 738: Detect completion via StreamCompleteEvent
    if isinstance(wrapped_event, StreamCompleteEvent) and wrapped_event.message:
        # This IS the close signal
        content = str(msg.content)
        # Finalize child session
        # Update parent container to "completed" state
```

### Information Already Available

| Information | Source | Notes |
|------------|--------|-------|
| Child session ID | `SubAgentEvent.child_session_id` | Same as `SpawnSessionStart.child_session_id` |
| Parent session ID | `SubAgentEvent.parent_session_id` | Same as `SpawnSessionStart.parent_session_id` |
| Final content | `StreamCompleteEvent.message.content` | Full output available |
| Token usage | `StreamCompleteEvent.message.usage` | Usage stats available |
| Cost info | `StreamCompleteEvent.message.cost_info` | Cost tracking available |
| Tool call ID | `SubAgentEvent.tool_call_id` | Correlates with spawn |

### Adding Close Event Would:

- **Duplicate information**: Same data available in `StreamCompleteEvent`
- **Increase complexity**: More event types to handle and document
- **Risk inconsistency**: Two ways to detect close (close event OR StreamCompleteEvent)

## Alternatives Considered

### Alternative 1: Extend RunStartedEvent

**Description**: Add `is_spawn`, `spawn_metadata`, etc. fields to `RunStartedEvent`.

| Criterion | Evaluation |
|-----------|------------|
| **Semantic Fit** | Poor - `RunStartedEvent` is internally emitted by the child agent, not at spawn time from parent |
| **Implementation** | Complex - would require passing spawn context into child agent |
| **Protocol Adapter** | Still reactive detection needed |
| **Verdict** | **Rejected** - semantic mismatch, doesn't solve the explicit signaling problem |

### Alternative 2: Add Metadata to SubAgentEvent

**Description**: Add `first_event: bool`, `spawn_metadata` fields to `SubAgentEvent`.

| Criterion | Evaluation |
|-----------|------------|
| **Semantic Fit** | Partial - still requires reactive detection on first SubAgentEvent |
| **Implementation** | Simple - no new event type |
| **Protocol Adapter** | Still reactive - cannot preemptively create session before first content event |
| **Verdict** | **Rejected** - doesn't provide explicit signal before content events begin |

### Alternative 3: The Chosen Approach (SpawnSessionStart)

| Criterion | Evaluation |
|-----------|------------|
| **Semantic Fit** | Excellent - explicit "spawn created" signal |
| **Implementation** | Moderate - one new event type, single emission point |
| **Protocol Adapter** | Proactive - can create session upfront with full metadata |
| **Verdict** | **Selected** - best trade-off between explicitness and simplicity |

## Implementation Plan

**Estimated Timeline**: ~5-6 days (revised based on complexity analysis)

| Phase | Description | Effort | Risk |
|-------|-------------|--------|------|
| 1 | Event Definition | 0.5 day | Low |
| 2 | Subagent Tools Integration | 1 day | Medium |
| 3 | Event Processor Update | 2 days | Medium |
| 4 | ACP Converter Update | 1 day | Low |
| 5 | Testing | 1-2 days | Medium |
| 6 | Storage Integration | 0.5 day | Low |
| **Total** | | **5.5 - 6.5 days** | |

### Phase 1: Event Definition (0.5 day)

**File**: `src/wolfharness/agents/events/events.py`

```python
@dataclass(kw_only=True)
class SpawnSessionStart:
    child_session_id: str
    parent_session_id: str
    tool_call_id: str | None
    spawn_mechanism: Literal["sync", "async_worker", "manual"]
    source_name: str
    source_type: Literal["agent", "team_parallel", "team_sequential"]
    depth: int = 1
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)
    event_kind: Literal["spawn_session_start"] = "spawn_session_start"

# Update RichAgentStreamEvent union
type RichAgentStreamEvent[OutputDataT] = (
    # ... existing events ...
    | SpawnSessionStart
)
```

### Phase 2: Subagent Tools Integration (1 day)

**File**: `src/wolfharness_toolsets/builtin/subagent_tools.py`

Update `_stream_task()` function:

```python
async def _stream_task(
    ctx: AgentContext,
    source_name: str,
    source_type: Literal["agent", "team_parallel", "team_sequential"],
    stream: AsyncIterator[RichAgentStreamEvent[Any]],
    *,
    prompt: str,  # NEW PARAMETER
    async_mode: bool = False,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Stream a task with SpawnSessionStart event."""
    
    # Generate session IDs
    child_session_id = identifier.ascending("session")
    parent_session_id = ctx.node.session_id
    tool_call_id = ctx.tool_call_id
    
    # Calculate depth (increment from parent if available)
    depth = 1
    if hasattr(ctx, "current_depth"):
        depth = ctx.current_depth + 1
    
    # EMIT: SpawnSessionStart (before any content)
    start_event = SpawnSessionStart(
        child_session_id=child_session_id,
        parent_session_id=parent_session_id,
        tool_call_id=tool_call_id,
        spawn_mechanism="async_worker" if async_mode else "sync",
        source_name=source_name,
        source_type=source_type,
        depth=depth,
        description=f"Run {source_name} task" + (f" (async: {task_id})" if async_mode else ""),
        metadata={
            "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "task_id": task_id,
            "max_depth": 5,  # Protocol-level depth limitation
        } if async_mode else {},
    )
    await ctx.events.emit_event(start_event)
    
    # Stream wrapped events...
    async for event in stream:
        subagent_event = SubAgentEvent(
            source_name=source_name,
            source_type=source_type,
            event=event,
            depth=depth,
            child_session_id=child_session_id,
            parent_session_id=parent_session_id,
            tool_call_id=tool_call_id,
        )
        await ctx.events.emit_event(subagent_event)
    
    # Return existing result
    return {"output": final_content, "metadata": {"sessionId": child_session_id}}
```

**Note on Background Workers**: Background workers (async_mode=True) currently write output to filesystem without emitting stream events. The RFC recommends:
- Emit `SpawnSessionStart` before spawning (for discovery)
- Document that real-time progress won't be available for background workers
- Consider adding polling-based completion events in future

### Phase 3: Event Processor Update (2 days)

**File**: `src/wolfharness_server/opencode_server/event_processor.py`

Add handler with duplicate session protection:

```python
from collections.abc import Iterator

async def process(self, event, ctx):
    match event:
        # ... existing cases ...
        case SpawnSessionStart():
            async for e in self._process_spawn_start(event, ctx):
                yield e

async def _process_spawn_start(
    self,
    event: SpawnSessionStart,
    ctx: EventProcessorContext,
) -> AsyncIterator[Event]:
    """Handle SpawnSessionStart with duplicate guard."""
    # CHECK: Prevent duplicate session creation
    if event.child_session_id in self._child_contexts:
        logger.debug(f"Session {event.child_session_id} already exists, ignoring duplicate")
        return
    
    # Create child session
    await ctx.state.ensure_session(
        event.child_session_id,
        parent_id=event.parent_session_id,
    )
    # ... rest of creation logic ...

# KEEP fallback in _process_subagent_event
async def _process_subagent_event(self, subagent_event, ctx, depth=0):
    """Keep reactive fallback for backward compatibility."""
    child_session_id = subagent_event.child_session_id
    child_ctx = self._child_contexts.get(child_session_id)
    if child_ctx is None and child_session_id:
        # FALLBACK: Session not created via SpawnSessionStart
        logger.warning(f"Reactive fallback for {child_session_id}")
        await ctx.state.ensure_session(child_session_id, parent_id=ctx.session_id)
        # ... fallback creation ...
```

### Phase 4: ACP Converter Update (1 day)

OPTION A: Simple text output (backward compatible):

```python
async def convert(self, event):
    match event:
        case SpawnSessionStart(source_name=name, description=desc, spawn_mechanism=mech):
            icon = "🚀" if mech == "sync" else "⚡"
            yield AgentMessageChunk.text(f"\n{icon} **`{name}`**: {desc}\n")
```

OPTION B: Tool call representation (richer UI):

```python
async def convert(self, event):
    match event:
        case SpawnSessionStart():
            yield ToolCallStart(
                tool_call_id=f"spawn:{event.child_session_id}",
                title=f"Spawned: {event.source_name}",
                kind="other",
                status="in_progress",
                metadata={
                    "type": "spawn_start",
                    "child_session_id": event.child_session_id,
                    "spawn_mechanism": event.spawn_mechanism,
                    "depth": event.depth,
                }
            )
```

### Phase 5: Testing (1-2 days)

```python
async def test_spawn_session_start_before_content():
    """Verify SpawnSessionStart emits before SubAgentEvent."""
    async with AgentPool() as pool:
        agent = pool.get_agent("test_agent")
        events = [e async for e in agent.run_stream("Use task tool")]
        
        spawn_idx = events.index(next(e for e in events if isinstance(e, SpawnSessionStart)))
        subagent_idx = events.index(next(e for e in events if isinstance(e, SubAgentEvent)))
        assert spawn_idx < subagent_idx

async def test_duplicate_session_guard():
    """Verify duplicate SpawnSessionStart rejected."""
    ...
```

### Phase 6: Storage Integration (0.5 day)

Update storage layer to persist `SpawnSessionStart` for analytics:

```python
if isinstance(event, SpawnSessionStart):
    await self.store_event(
        session_id=event.parent_session_id,
        event_type="spawn_start",
        event_data={"child_session_id": event.child_session_id, ...}
    )
```

## Comparison: With vs Without SpawnSessionStart

### Without (Current - Reactive)
- Session created on first SubAgentEvent
- No metadata available upfront
- May receive content before session ready

### With (Proposed - Proactive)
- Session created before any content
- Rich metadata available for UI
- Self-documenting code

## Backward Compatibility

- **Consumers**: 100% backward compatible - ignored if not handled
- **Producers**: Emitting SpawnSessionStart doesn't break existing adapters
- **Migration**: Gradual adoption via fallback in EventProcessor

## Decision Record

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-14 | Single event (start only) | StreamCompleteEvent handles close |
| 2026-02-14 | `spawn_mechanism` values: sync, async_worker, manual | Clearer than task/worker/background |
| 2026-02-14 | `depth` as first-class field | Needed for capping, UI, analytics |
| 2026-02-14 | Duplicate session guard | Prevents race condition issues |
| 2026-02-14 | Timeline: 5-6 days | Revised from 3 days based on complexity |

## References

- [RFC-0013: Subagent Event Unification](./RFC-0013-subagent-event-unification.md)
- [AgentPool Events](./src/wolfharness/agents/events/events.py)
- [Subagent Tools](./src/wolfharness_toolsets/builtin/subagent_tools.py)
