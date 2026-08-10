# AgentPool run_stream() Break Bug Report

**Report Date**: 2026-03-24
**Resolution Date**: 2026-03-23
**Reporter**: AgentPool Simulation Framework Team
**Severity**: High - Breaks core simulation use case
**Status**: ✅ FIXED

---

## Summary

Breaking from `Agent.run_stream()` iteration causes critical errors due to CancelScope context switching issues. This prevents implementing simulation frameworks that need to pause agent execution mid-stream.

---

## Problem Description

### Expected Behavior

When breaking from an async generator like `run_stream()`:
```python
async for event in agent.run_stream("Hello"):
    if should_stop(event):
        break  # Should exit cleanly
# Continue normal execution
```

### Actual Behavior

Breaking triggers multiple errors:

```python
RuntimeError: Attempted to exit cancel scope in a different task than it was entered in
ValueError: <Token> was created in a different Context
RuntimeError: generator didn't stop after athrow()
CancelledError: Cancelled via cancel scope
```

---

## Root Cause Analysis

### Where the Bug Occurs

**File**: `/Users/yuchen.liu/src/yilab/iroot-llm/packages/wolfharness/src/wolfharness/agents/native_agent/agent.py`

**Lines**: 829-845

```python
# Problematic code pattern
async with (
    node.stream(agent_run.ctx) as stream,
    merge_queue_into_iterator(stream, self._event_queue) as merged,
):
    async for event in merged:
        yield event
        # ...
except GeneratorExit:
    # GeneratorExit is caught here, but cleanup fails
    self._cancelled = True
```

### Technical Details

1. **merge_queue_into_iterator creates tasks**: The utility function spawns separate tasks to merge streams
2. **CancelScope crosses task boundaries**: AnyIO's CancelScope is entered in one task but cleanup occurs in another
3. **ContextVar mismatch**: asyncio ContextVars are task-local; the merged iterator task has different context

### Call Stack Affected

```
agent.run_stream() 
  → _run_stream_once()
    → _stream_events()
      → agentlet.iter()  # pydantic-ai
        → async for node in agent_run
          → node.stream() 
            → merge_queue_into_iterator()  # <-- Problem here
              → Creates background task
                → CancelScope entered in task A
                → Cleanup attempted in task B (break context)
```

---

## Reproduction Steps

### Test Script Location

`/Users/yuchen.liu/src/yilab/iroot-llm/packages/wolfharness/tests/test_break_behavior.py`

### Minimal Reproduction

```python
import asyncio
from wolfharness import Agent
from pydantic_ai.models.test import TestModel

async def test_break():
    """Minimal reproduction of the break bug."""
    model = TestModel()
    
    async with Agent(
        name="test",
        model=model,
    ) as agent:
        try:
            async for event in agent.run_stream("Hello"):
                print(f"Received: {type(event).__name__}")
                break  # This triggers the bug
            
            print("After break - should reach here but may not")
            
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")
            raise

if __name__ == "__main__":
    asyncio.run(test_break())
```

### Observed Errors

1. **RuntimeError**: Cancel scope exit task mismatch
2. **ValueError**: Context token created in different context
3. **RuntimeError**: Generator didn't stop after athrow()
4. **CancelledError**: Cancelled via cancel scope

---

## Impact

### Affected Use Cases

| Use Case | Impact |
|----------|--------|
| **Simulation Framework** | ❌ Cannot pause on elicitation detection |
| **Early Stream Termination** | ❌ All break scenarios affected |
| **Interactive Interrupt** | ⚠️ May have issues with Ctrl+C handling |
| **Timeout Handling** | ⚠️ May not clean up properly on timeout |

### Current Workarounds

**None available** - Any break from run_stream is affected.

**Partial Workaround**: Use `run()` instead of `run_stream()`:
```python
# Instead of streaming with break
result = await agent.run("Hello")  # Complete execution, no streaming
```

But this loses the ability to interrupt mid-execution.

---

## Investigation Tasks

- [ ] Confirm bug exists with latest AgentPool code
- [ ] Isolate whether issue is in AgentPool or pydantic-ai
- [ ] Test with different AnyIO backends (asyncio vs trio)
- [ ] Verify if SingleTaskBeater (subprocess) has same issue
- [ ] Create minimal reproduction without AgentPool wrapper
- [ ] Document safe break patterns if any exist

### Deep Dive Areas

1. **merge_queue_into_iterator** (`/Users/yuchen.liu/src/yilab/iroot-llm/packages/wolfharness/src/wolfharness/utils/streams.py`)
   - How are tasks spawned?
   - How is cancellation propagated?
   - Can we make scope exit task-aware?

2. **AnyIO CancelScope** compatibility
   - Check AnyIO documentation for cross-task scope patterns
   - Review if there's a supported way to handle this

3. **pydantic-ai iter() behavior**
   - Does raw pydantic-ai `agent.iter()` have same issue?
   - Is AgentPool's wrapper introducing the problem?

---

## Related Code

### Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `wolfharness/agents/native_agent/agent.py` | 808-895 | Native agent stream with GeneratorExit handling |
| `wolfharness/agents/native_agent/agent.py` | 829-845 | merge_queue_into_iterator usage |
| `wolfharness/utils/streams.py` | ~ | merge_queue_into_iterator implementation |
| `wolfharness/agents/base_agent.py` | 566-648 | run_stream() top-level loop |

### Key Code Segments

**Native Agent GeneratorExit Handling** (agent.py:839-845):
```python
except GeneratorExit:
    # Consumer stopped iteration early (e.g., by break)
    # Avoid re-raising to prevent cleanup in wrong context
    self._cancelled = True
    self.log.debug("GeneratorExit caught in node stream, cancelling gracefully")
    # Do not re-raise - let finally blocks clean up normally
```

*Note: The attempt to "not re-raise" doesn't prevent the deeper CancelScope issue.*

---

## Potential Solutions

### Option 1: Fix merge_queue_into_iterator

Make the merged iterator cleanup aware of the calling task context.

**Effort**: High
**Risk**: May affect other streaming uses

### Option 2: Avoid merge_queue_into_iterator in simulation paths

Create a simplified streaming path that doesn't merge queues.

**Effort**: Medium
**Risk**: Code duplication

### Option 3: Use pydantic-ai iter() directly

Bypass AgentPool's stream wrapper, use pydantic-ai's native iteration.

**Effort**: Medium
**Risk**: Loses AgentPool features (hooks, event handlers, etc.)

### Option 4: Create SimulationRun abstraction

New class that doesn't use run_stream at all, uses manual node iteration.

**Effort**: Medium
**Risk**: New API surface, maintenance burden

See: `docs/rfc/RFC-001-pause-resume-iteration.md` for detailed design.

---

## References

- **Simulation Framework RFC**: `docs/rfcs/draft/RFC-0018-simulation-framework.md`
- **Pause/Resume Iteration RFC**: `docs/rfc/RFC-001-pause-resume-iteration.md`  
- **Background Task - Break Validation**: `bg_f38301c0`
- **Background Task - Iter Design**: `bg_1a7733d7`

---

## Resolution

### Fix Summary

The bug was fixed by isolating the entire `agentlet.iter()` iteration in a background task. This ensures that when the consumer breaks from the async iteration:

1. **Consumer task**: Handles the `break` statement without triggering cleanup in pydantic-ai's context managers
2. **Background task**: Runs `agentlet.iter()` and pydantic-ai's CancelScope/TaskGroup in its own task context
3. **Communication**: Events are passed via an `asyncio.Queue` between the tasks
4. **Cleanup**: When consumer breaks, we signal the background task to stop gracefully and let it clean up its own context managers

### Changes Made

1. **`src/wolfharness/utils/streams.py`**: Enhanced `merge_queue_into_iterator` with:
   - `shutdown_event` for cooperative cancellation
   - `except GeneratorExit` handler to signal graceful shutdown
   - Task-aware cleanup that uses `asyncio.shield` to prevent CancelScope issues

2. **`src/wolfharness/agents/native_agent/agent.py`**: Refactored `_stream_events()` to:
   - Run the entire `agentlet.iter()` iteration in a background task
   - Yield events via a queue from the consumer task
   - Properly signal cancellation and cleanup

### Test Results

All 8 break behavior tests now pass:
- ✅ Simple break after N events
- ✅ Exception handling (no exceptions propagate to user)
- ✅ Conversation history after break
- ✅ **Subsequent run after break** (was failing, now works)
- ✅ Interrupt vs break
- ✅ Safe pattern - complete consumption
- ✅ Tool detection without break
- ✅ Partial text collection

### Impact

The Simulation Framework can now:
- ✅ Break from `run_stream()` to pause on elicitation detection
- ✅ Resume agent execution with subsequent `run_stream()` calls
- ✅ Avoid CancelScope/ContextVar errors

---

**This bug has been resolved. The Simulation Framework implementation is unblocked.**
