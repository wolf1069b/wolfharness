# RFC-0021: Agent Concurrent Execution Safety

## Header Metadata

---
rfc_id: RFC-0021
title: Agent Concurrent Execution Safety
status: ACCEPTED
author: yuchen.liu
reviewers: []
created: 2025-04-05
last_updated: 2026-04-05
decision_date:
related_documents:
  - RFC-0021-PRE-FLIGHT-ANALYSIS.md (State inventory and audit)
  - tests/agents/test_concurrent_safety.py (Test suite)
  - tests/agents/run_concurrent_tests.py (Quick validation)
---

## 1. Overview

### 1.1 Summary

This RFC proposes a fundamental redesign of AgentPool's agent execution model to support **safe concurrent calls** to the same agent instance. The current implementation shares instance-level mutable state (`_cancelled`, `_current_stream_task`, `_event_queue`, `_injection_manager`) across concurrent `run_stream()` calls, causing race conditions, premature task termination, and data corruption.

### 1.2 Why This Matters Now

**Critical Production Issue**: In multi-agent delegation scenarios where a main agent spawns 3+ concurrent subagent tasks via `new_task`, we observe:
- First 2 tasks appear "interrupted" mid-execution
- Only the last task completes normally
- 56% failure rate in concurrent scenarios (based on test reports)

This blocks production deployment of parallel agent teams and limits scalability.

### 1.3 Expected Outcome

After implementation:
- **True concurrency**: Multiple `run_stream()` calls to the same agent instance execute independently
- **No shared state pollution**: Each call has isolated execution context
- **Backward compatibility**: Serial calls continue to work unchanged
- **Performance**: Parallel execution reduces wall-clock time for multi-agent workflows

## 2. Background & Context

### 2.1 Current Architecture

```python
class BaseAgent:
    def __init__(self, ...):
        # Instance-level mutable state (PROBLEM)
        self._cancelled = False                    # Shared across calls
        self._current_stream_task = None           # Overwritten per call
        self._event_queue = asyncio.Queue()        # Shared queue
        self._injection_manager = PromptInjectionManager()  # Shared state

    async def run_stream(self, prompts, deps=None):
        self._cancelled = False
        self._current_stream_task = asyncio.current_task()
        # ... all calls mutate the same instance state
```

### 2.2 Problem Evidence

**Test Results** (`test_agent_state_sharing.py`):
```
[B] 设置 _cancelled=True
[A] ⚠️ 检测到 _cancelled=True，提前终止！
```

**Code Analysis** (`base_agent.py:767`):
```python
if self._cancelled:  # ← Any concurrent call can set this!
    break
```

**Existing Bug in `finally` Block**:
```python
# native_agent.py:914-917
finally:
    iteration_done.set()
    self._cancelled = True  # ← BUG: Sets cancelled even on normal completion!
```

This is a semantic bug: `_cancelled` should only be set when a run is actually cancelled, not when it completes normally. This bug exacerbates the concurrent execution problem.

### 2.3 Pre-Flight Analysis

A comprehensive state inventory and codebase audit has been completed. See [RFC-0021-Pre-Flight-Analysis.md](./RFC-0021-PRE-FLIGHT-ANALYSIS.md) for full details.

**Key Findings**:

| State Field | Access Count | Risk Level | Migration Priority |
|-------------|--------------|------------|-------------------|
| `_cancelled` | 15+ locations | **Critical** | P0 |
| `_current_stream_task` | 8 locations | **Critical** | P0 |
| `_event_queue` | 6 locations | **Critical** | P1 |
| `_injection_manager` | 9 locations | **High** | P1 |
| `_background_task` | 4 locations | Medium | P2 |

**Subclass Impact Assessment**:
- `NativeAgent`: High risk (direct state access in 9+ locations)
- `ACPAgent`: Medium risk (event queue access)
- `AGUIAgent`: Medium risk (cancellation checks)
- `ClaudeCodeAgent`: Medium risk (event handling)
- `CodexAgent`: Requires audit

**Do NOT Migrate** (Intentionally Shared):
- `_formatted_system_prompt`: Represents agent's shared personality
- `_internal_fs`: Shared filesystem is a design feature

### 2.4 Glossary

### 2.3 Glossary

| Term | Definition |
|------|------------|
| **Agent Instance** | Single `BaseAgent` object that can receive multiple `run_stream()` calls |
| **Run Context** | Per-execution isolated state container (proposed solution) |
| **Concurrent Safety** | Ability to safely execute multiple async operations simultaneously without interference |
| **Shared State Pollution** | When one execution modifies state that affects other concurrent executions |

### 2.4 Related Work

- RFC-0015: Session tracking for subagent events (complementary to this RFC)
- Issue: `new_task` concurrent delegation returns incorrect results

## 3. Problem Statement

### 3.1 Specific Problem

When a main agent executes 3 concurrent `new_task` calls to the same subagent instance:

1. **State Overwrite**: Each call overwrites `_current_stream_task`, losing reference to previous tasks
2. **Cancellation Propagation**: When one task completes and sets `_cancelled = True`, all concurrent tasks check this flag and terminate early
3. **Event Queue Confusion**: All events from multiple subagents flow into the same `_event_queue`, causing cross-pollution
4. **Injection Manager Corruption**: Shared `PromptInjectionManager` state causes prompts intended for one call to affect others

### 3.2 Evidence

| Metric | Serial Execution | Concurrent Execution |
|--------|-----------------|---------------------|
| Success Rate | 100% (9/9) | 44% (4/9) |
| Early Termination | 0% | 56% (5/9) |
| Correct Result Capture | 100% | ~60% |

### 3.3 Impact of Not Solving

- **Blocked Feature**: Parallel agent teams cannot be safely deployed
- **Workaround Cost**: Users must serialize agent calls, losing performance benefits
- **Reliability Risk**: Race conditions in production cause unpredictable failures
- **Scalability Ceiling**: Cannot leverage async concurrency for multi-agent workflows

## 4. Goals & Non-Goals

### 4.1 Goals (In Scope)

1. **Primary**: Enable safe concurrent `run_stream()` calls to the same agent instance
2. **Secondary**: Isolate per-execution state (`_cancelled`, `_current_stream_task`, `_event_queue`, `_injection_manager`)
3. **Secondary**: Maintain backward compatibility for serial execution
4. **Secondary**: Provide clear migration path for existing code

### 4.2 Non-Goals (Out of Scope)

1. **Not**: Changing the AgentPool architecture fundamentally
2. **Not**: Supporting concurrent calls to different agent instances (already works)
3. **Not**: Modifying the underlying LLM provider concurrency model
4. **Not**: Addressing thread-safety (asyncio-only for now)
5. **Not**: Adding distributed execution support

### 4.3 Success Criteria

- [ ] 3+ concurrent `run_stream()` calls to same agent complete successfully
- [ ] No shared state pollution (verified by tests)
- [ ] Serial execution performance unchanged (±5%)
- [ ] All existing tests pass without modification
- [ ] New concurrent safety tests added and passing

## 5. Evaluation Criteria

| Criterion | Weight | Description | Measurement |
|-----------|--------|-------------|-------------|
| **Concurrent Safety** | Critical | Eliminate race conditions | Test: 100 concurrent calls complete correctly |
| **Backward Compatibility** | High | Existing code continues to work | All existing tests pass |
| **Implementation Complexity** | Medium | Reasonable effort and risk | Estimated dev days |
| **Performance** | Medium | No significant overhead | Benchmark: ±10% of baseline |
| **Maintainability** | Medium | Code remains understandable | Code review approval |
| **Debuggability** | Low | Easy to diagnose issues | Can trace per-call state |

## 6. Options Analysis

### Option 1: Serialization via Lock (Quick Fix)

**Description**: Use `asyncio.Lock` to serialize concurrent calls to `run_stream()`.

```python
class BaseAgent:
    def __init__(self, ...):
        self._stream_lock = asyncio.Lock()

    async def run_stream(self, ...):
        async with self._stream_lock:
            # Original implementation
```

**Advantages**:
- Minimal code change (~3 lines)
- Immediate fix with no state isolation needed
- Very low risk of regression

**Disadvantages**:
- Loses all concurrency benefits
- Performance degrades to serial execution
- Does not address root cause (shared state)
- Blocks on slow LLM calls

**Evaluation**:
| Criterion | Score | Notes |
|-----------|-------|-------|
| Concurrent Safety | ⚠️ | Safe but not concurrent |
| Backward Compatibility | ✅ | Perfect |
| Implementation Complexity | ✅ | 1 hour |
| Performance | ❌ | Serial execution |
| Maintainability | ✅ | Simple |
| Debuggability | ✅ | Same as today |

**Effort Estimate**: 0.5-1 day

**Risk Assessment**: Very Low - minimal code change

---

### Option 2: Per-Call Execution Context (Recommended)

**Description**: Create isolated `AgentRunContext` for each `run_stream()` call, moving mutable state from instance to call level.

```python
@dataclass
class AgentRunContext:
    cancelled: bool = False
    current_task: asyncio.Task | None = None
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    injection_manager: PromptInjectionManager = field(default_factory=PromptInjectionManager)
    session_id: str = field(default_factory=lambda: str(uuid4()))
    deps: Any = None

class BaseAgent:
    async def run_stream(self, ...):
        run_ctx = AgentRunContext(deps=deps)
        run_ctx.current_task = asyncio.current_task()
        async for event in self._run_with_context(run_ctx, ...):
            yield event
```

**Advantages**:
- True concurrent execution
- Clear isolation boundaries
- Easy to understand and debug
- Backward compatible for serial calls
- Type-safe with dataclass

**Disadvantages**:
- Moderate code changes (~200 lines)
- Need to pass context through call chain
- May require updates to subclasses

**Evaluation**:
| Criterion | Score | Notes |
|-----------|-------|-------|
| Concurrent Safety | ✅ | Full isolation |
| Backward Compatibility | ✅ | API unchanged |
| Implementation Complexity | ⚠️ | 2-3 days |
| Performance | ✅ | Parallel execution |
| Maintainability | ✅ | Clear separation |
| Debuggability | ✅ | Per-call traceable |

**Effort Estimate**: 3-5 days

**Risk Assessment**: Low-Medium - touches core execution path

---

### Option 3: Async Context Variables

**Description**: Use Python's `contextvars` for implicit state isolation.

```python
import contextvars

_cancelled_var: contextvars.ContextVar[bool] = contextvars.ContextVar('cancelled', default=False)

class BaseAgent:
    async def run_stream(self, ...):
        _cancelled_var.set(False)
        # Access via _cancelled_var.get()
```

**Advantages**:
- Elegant, Pythonic solution
- No need to pass context explicitly
- Automatic isolation per async call

**Disadvantages**:
- Implicit state harder to trace and debug
- Team must understand `contextvars`
- Can be surprising behavior
- Testing requires context setup

**Evaluation**:
| Criterion | Score | Notes |
|-----------|-------|-------|
| Concurrent Safety | ✅ | Full isolation |
| Backward Compatibility | ✅ | API unchanged |
| Implementation Complexity | ⚠️ | 2-3 days |
| Performance | ✅ | Parallel execution |
| Maintainability | ⚠️ | Implicit complexity |
| Debuggability | ❌ | Hard to trace |

**Effort Estimate**: 3-4 days

**Risk Assessment**: Medium - team learning curve and debugging complexity

---

### Option 4: Agent Instance Pool

**Description**: Create multiple agent instances, each handling one concurrent call.

```python
class AgentPool:
    def __init__(self, factory, max_instances=10):
        self._instances = [factory() for _ in range(max_instances)]
        self._semaphore = asyncio.Semaphore(max_instances)

    async def run_stream(self, ...):
        async with self._semaphore:
            agent = self._acquire()
            async for event in agent.run_stream(...):
                yield event
            self._release(agent)
```

**Advantages**:
- Maximum isolation (separate instances)
- No changes to BaseAgent internals
- Resource control via semaphore

**Disadvantages**:
- High memory overhead (N instances × agent state)
- Complex lifecycle management
- Configuration complexity
- No shared state benefits (conversation history)

**Evaluation**:
| Criterion | Score | Notes |
|-----------|-------|-------|
| Concurrent Safety | ✅ | Maximum isolation |
| Backward Compatibility | ⚠️ | API change needed |
| Implementation Complexity | ❌ | 1-2 weeks |
| Performance | ⚠️ | Memory overhead |
| Maintainability | ❌ | Complex lifecycle |
| Debuggability | ✅ | Instance-per-call |

**Effort Estimate**: 1-2 weeks

**Risk Assessment**: High - architectural change with many moving parts

---

## 7. Recommendation

### 7.1 Recommended Option: Option 2 (Per-Call Execution Context)

**Justification**:

1. **Best Balance**: Achieves true concurrency with reasonable implementation effort
2. **Clear Ownership**: Explicit context passing makes data flow visible and debuggable
3. **Team Accessibility**: Easier to understand than implicit context variables
4. **Incremental Migration**: Can be implemented in phases without breaking changes
5. **Future-Proof**: Provides foundation for additional per-call features

**Trade-offs Accepted**:
- Moderate implementation effort (3-5 days)
- Requires passing context through call chain
- May need updates to subclasses

**Alternatives Considered**:
- Option 1 (Lock) rejected: Does not achieve concurrency goal
- Option 3 (ContextVars) rejected: Debugging complexity outweighs benefits
- Option 4 (Pool) rejected: Too complex for current needs

### 7.2 Decision Rationale

Based on evaluation criteria, Option 2 scores highest overall:

| Criterion | Option 1 | Option 2 | Option 3 | Option 4 |
|-----------|----------|----------|----------|----------|
| Concurrent Safety | ⚠️ | ✅ | ✅ | ✅ |
| Backward Compatibility | ✅ | ✅ | ✅ | ⚠️ |
| Implementation Complexity | ✅ | ⚠️ | ⚠️ | ❌ |
| Performance | ❌ | ✅ | ✅ | ⚠️ |
| Maintainability | ✅ | ✅ | ⚠️ | ❌ |
| **Overall** | ❌ | ✅ | ⚠️ | ❌ |

## 8. Technical Design

### 8.1 Architecture

```
┌─────────────────────────────────────────────────────┐
│                  BaseAgent (Instance)               │
│  ┌───────────────────────────────────────────────┐  │
│  │  Shared State (Immutable/Per-Instance)        │  │
│  │  - name, description, model_name              │  │
│  │  - tools (ToolManager)                        │  │
│  │  - _internal_fs (IsolatedMemoryFileSystem)    │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  Call 1: run_stream()                         │  │
│  │  ┌─────────────────────────────────────────┐  │  │
│  │  │ AgentRunContext 1                      │  │  │
│  │  │ - cancelled: False                      │  │  │
│  │  │ - event_queue: Queue1                   │  │  │
│  │  │ - injection_manager: Manager1           │  │  │
│  │  │ - session_id: "uuid-1"                  │  │  │
│  │  └─────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  Call 2: run_stream()                         │  │
│  │  ┌─────────────────────────────────────────┐  │  │
│  │  │ AgentRunContext 2                      │  │  │
│  │  │ - cancelled: False                      │  │  │
│  │  │ - event_queue: Queue2                   │  │  │
│  │  │ - injection_manager: Manager2           │  │  │
│  │  │ - session_id: "uuid-2"                  │  │  │
│  │  └─────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### 8.2 Data Model

```python
@dataclass
class AgentRunContext:
    """
    Per-execution isolated context for concurrent safety.
    
    Each run_stream() call creates a new AgentRunContext instance,
    ensuring no shared mutable state between concurrent calls.
    """
    # Cancellation state
    cancelled: bool = False
    
    # Task reference
    current_task: asyncio.Task | None = None
    
    # Event queue (isolated per call)
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    
    # Prompt injection state (isolated per call)
    injection_manager: PromptInjectionManager = field(
        default_factory=PromptInjectionManager
    )
    
    # Session identification
    session_id: str = field(default_factory=lambda: str(uuid4()))
    
    # Dependencies passed to run_stream()
    deps: Any = None
    
    # Additional per-call state as needed
    start_time: float = field(default_factory=time.perf_counter)
```

### 8.3 API Changes

**No public API changes** - `run_stream()` signature remains identical:

```python
async def run_stream(
    self,
    *prompts: PromptCompatible,
    deps: TDeps | None = None,
    ...
) -> AsyncIterator[RichAgentStreamEvent[TResult]]:
    """Execute agent with concurrent safety.
    
    Multiple concurrent calls to run_stream() are now safe and
    execute in parallel with isolated state.
    """
```

**Internal Changes**:

```python
# Before
async def _run_stream_once(self, prompts, ...):
    self._cancelled = False
    if self._cancelled:  # Check shared state
        break

# After
async def _run_stream_once(self, run_ctx: AgentRunContext, prompts, ...):
    run_ctx.cancelled = False
    if run_ctx.cancelled:  # Check isolated state
        break
```

### 8.4 Security Considerations

1. **Isolation**: Per-call event queues prevent information leakage between concurrent executions
2. **Resource Limits**: Each call has independent injection manager, preventing prompt injection attacks across calls
3. **Audit Trail**: Per-call session_id enables better logging and tracing

## 9. Implementation Plan

### Phase 0: Pre-Flight & Bug Fix (Day 0)

**Prerequisite Tasks** (must complete before migration):
1. **Review Pre-Flight Analysis**: Study `RFC-0021-PRE-FLIGHT-ANALYSIS.md` for complete state inventory
2. **Fix `finally` Block Bug**: Correct semantic error in `native_agent.py:917`
   ```python
   # Before (BUG):
   finally:
       self._cancelled = True  # Always sets cancelled
   
   # After (FIX):
   finally:
       iteration_done.set()
       # Only set cancelled if actually cancelled
       if task.cancelled():
           self._cancelled = True
   ```
3. **Establish Test Baseline**: Run pre-flight test suite
   ```bash
   python tests/agents/run_concurrent_tests.py
   ```
4. **Verify Test Failure**: Ensure concurrent tests fail before fix (proves test validity)

**Files Modified**:
- `src/wolfharness/agents/native_agent.py` (fix finally block)

**Deliverable**: 
- Existing bug fixed
- Failing test baseline established
- State inventory confirmed

**Rollback**: `git revert` on single commit

### Phase 1: Core Context Creation (Days 1-2)

**Tasks**:
1. Create `AgentRunContext` dataclass in `wolfharness/agents/context.py`
2. Add `run_ctx` parameter to internal methods
3. Migrate `_cancelled` and `_current_stream_task` to context
4. Run tests after each migration step

**Files Modified**:
- `src/wolfharness/agents/context.py` (add AgentRunContext)
- `src/wolfharness/agents/base_agent.py` (core changes)

**Deliverable**: Basic concurrent calls work for simple scenarios

**Rollback**: Revert changes, restore from git

### Phase 2: Full State Migration (Days 3-4)

**Tasks**:
1. Migrate `_event_queue` to context (most complex - cross-cutting)
2. Migrate `_injection_manager` to context
3. Update event emitter to use context
4. Ensure proper cleanup in finally blocks
5. Run full test suite after each step

**Files Modified**:
- `src/wolfharness/agents/base_agent.py`
- `src/wolfharness/agents/events/event_emitter.py`
- `src/wolfharness/agents/context.py`

**Deliverable**: All mutable state isolated

**Rollback**: Revert changes, restore from git

### Phase 3: Testing & Validation (Days 5-7)

**Tasks**:
1. Run comprehensive concurrent safety test suite (`tests/agents/test_concurrent_safety.py`)
2. Run full existing test suite to check for regressions
2. Run full test suite to ensure no regressions
3. Performance benchmarks (serial vs concurrent)
4. Documentation updates

**Files Modified**:
- `tests/agents/test_concurrent_safety.py` (new)
- `docs/` updates

**Deliverable**: Production-ready implementation

**Rollback**: Revert changes, restore from git

### Dependencies

- None external
- Internal: All changes within `wolfharness.agents` module

## 10. Open Questions

1. **Conversation History**: Should `conversation` be shared between concurrent calls or isolated? Currently leaning toward shared (represents agent's accumulated knowledge).

2. **Event Propagation**: When subagent events are forwarded to parent via `emit_event()`, should they be tagged with the run context session_id for filtering?

3. **Subclass Compatibility**: Do any subclasses rely on accessing `_cancelled` or other instance state directly? Need to audit.

4. **Performance Benchmarks**: What are acceptable overhead thresholds for context creation per call?

## 11. Decision Record

**Status**: DRAFT (awaiting review)

**Decision**: TBD

**Date**: TBD

**Approvers**: TBD

**Key Discussion Points**:
- Option 2 selected for balance of safety and maintainability
- Option 3 (ContextVars) considered but rejected for debugging complexity
- Implementation to be done in 3 phases for risk mitigation

**Conditions on Approval**:
- [ ] At least 2 code reviewers approve
- [ ] All existing tests pass
- [ ] New concurrent safety tests demonstrate 100 concurrent calls succeed
- [ ] Performance benchmark shows <10% overhead for serial calls
- [ ] Documentation updated

---

## Appendix A: Migration Guide for Subclasses

If you have custom agent subclasses that access instance state:

### Before
```python
class MyAgent(BaseAgent):
    async def _custom_method(self):
        if self._cancelled:  # Accessing instance state
            return
```

### After
```python
class MyAgent(BaseAgent):
    async def _custom_method(self, run_ctx: AgentRunContext):
        if run_ctx.cancelled:  # Accessing context state
            return
```

## Appendix B: Test Strategy

### Test Suite Location

Complete test suite available at: `tests/agents/test_concurrent_safety.py`

Quick validation script: `tests/agents/run_concurrent_tests.py`

### Test Categories

#### 1. Baseline Tests (Must pass before and after)
- `test_serial_execution_baseline` - Serial execution still works
- `test_single_call_completion` - Single call completes normally

#### 2. Concurrent Isolation Tests (Primary validation)
- `test_concurrent_calls_complete` - All concurrent calls finish
- `test_concurrent_event_isolation` - Events don't cross-contaminate
- `test_concurrent_cancellation_isolation` - Cancellation is isolated
- `test_concurrent_event_queue_isolation` - Queue isolation verified

#### 3. Stress Tests
- `test_10_concurrent_calls` - 10-way concurrency
- `test_rapid_fire_concurrent_calls` - Rapid-fire launches

#### 4. Performance Tests
- `test_serial_performance_baseline` - No regression in serial mode
- `test_concurrent_performance` - Parallel speedup >1.5x

### Running Tests

```bash
# Pre-flight validation (before implementation)
python tests/agents/run_concurrent_tests.py

# Full test suite (after implementation)
pytest tests/agents/test_concurrent_safety.py -v

# With coverage
pytest tests/agents/test_concurrent_safety.py --cov=src/wolfharness/

# Stress test only
pytest tests/agents/test_concurrent_safety.py -m slow -v
```

### Example Test

```python
# Test: test_concurrent_isolation
async def test_concurrent_event_isolation():
    """Each concurrent call must have isolated event streams."""
    agent = Agent(name="test", model="test")
    
    async def run_with_marker(marker: str):
        events = []
        async for event in agent.run_stream(f"Task {marker}"):
            events.append(event)
        return events
    
    # Run 10 concurrent tasks
    results = await asyncio.gather(*[run_with_marker(f"M{i}") for i in range(10)])
    
    # Each result should have complete event sequence
    for i, events in enumerate(results):
        assert any(f"M{i}" in str(e) for e in events), f"Task {i} events corrupted"
```

## Appendix C: References

### Related Documents

1. **RFC-0021-Pre-Flight-Analysis.md**  
   Complete state inventory, subclass audit, and code path analysis.  
   Location: `docs/rfcs/draft/RFC-0021-PRE-FLIGHT-ANALYSIS.md`

2. **Test Suite**  
   Comprehensive concurrent safety tests.  
   Location: `tests/agents/test_concurrent_safety.py`

3. **Quick Validation Script**  
   Pre-flight test runner for rapid validation.  
   Location: `tests/agents/run_concurrent_tests.py`

4. **Related RFCs**  
   - RFC-0015: Session tracking for subagent events (complementary)

### Code Locations

Key files referenced in this RFC:
- `src/wolfharness/agents/base_agent.py` - BaseAgent implementation
- `src/wolfharness/agents/native_agent/agent.py` - NativeAgent implementation
- `src/wolfharness/agents/context.py` - AgentContext and (new) AgentRunContext
- `src/wolfharness/agents/events/event_emitter.py` - Event emission

---

**End of RFC-0021**

## Appendix B: Implementation Summary

### Status: COMPLETED ✅

All phases of RFC-0021 have been successfully implemented:

- **Phase 0**: Finally block bug fixed in NativeAgent
- **Wave 1**: AgentRunContext created, _cancelled and _current_stream_task migrated
- **Wave 2**: _event_queue and _injection_manager migrated, cleanup verified
- **Wave 3**: All tests passing (9/9 concurrent safety tests)

### Test Results

```
tests/agents/test_concurrent_safety.py::test_serial_execution_baseline PASSED
tests/agents/test_concurrent_safety.py::test_single_call_completion PASSED
tests/agents/test_concurrent_safety.py::test_concurrent_calls_complete PASSED
tests/agents/test_concurrent_safety.py::test_concurrent_event_isolation PASSED
tests/agents/test_concurrent_safety.py::test_concurrent_cancellation_isolation PASSED
tests/agents/test_concurrent_safety.py::test_concurrent_event_queue_isolation PASSED
tests/agents/test_concurrent_safety.py::test_serial_performance_baseline PASSED
tests/agents/test_concurrent_safety.py::test_concurrent_performance PASSED
tests/agents/test_concurrent_safety.py::test_native_agent_concurrent PASSED
```

### Files Modified

- `src/wolfharness/agents/context.py` - AgentRunContext dataclass
- `src/wolfharness/agents/base_agent.py` - State migration to context
- `src/wolfharness/agents/native_agent/agent.py` - Bug fixes and context usage
- `src/wolfharness/agents/events/event_emitter.py` - Context-based event queue
- All agent subclasses updated for context compatibility

### Migration Guide for Custom Subclasses

If you have custom agent subclasses that access mutable state:

1. **Update method signatures** to accept `run_ctx: AgentRunContext`:
   ```python
   async def _stream_events(self, run_ctx: AgentRunContext, ...)
   async def _interrupt(self, run_ctx: AgentRunContext | None = None)
   ```

2. **Access state via context** instead of instance:
   ```python
   # Before
   if self._cancelled:
       
   # After
   if run_ctx.cancelled:
   ```

3. **Use context for event queue**:
   ```python
   # Before
   await self._event_queue.put(event)
   
   # After
   await run_ctx.event_queue.put(event)
   ```

4. **Pass context through call chain** when calling parent methods.

