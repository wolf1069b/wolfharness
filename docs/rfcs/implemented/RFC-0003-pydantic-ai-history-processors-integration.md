---
rfc_id: RFC-0003
title: PydanticAI History Processors Integration
status: DRAFT
author: Sisyphus
reviewers:
  - name: Metis
    status: pending
created: 2026-02-02
last_updated: 2026-02-02
decision_date:
related_prds: []
related_rfcs: [RFC-0002]
---

# RFC-0003: PydanticAI History Processors Integration

## Overview

This RFC proposes adding support for pydantic-ai's `history_processors` mechanism to wolfharness's native agent configuration. History processors are callables that transform message history before it's sent to the model, enabling advanced conversation management patterns like context-aware filtering, summarization, token budget management, and custom message selection logic.

The integration aims to provide a clean pass-through to pydantic-ai's native history processing capabilities with minimal code changes, allowing users to leverage the full power of pydantic-ai's history processor ecosystem directly from wolfharness's YAML configuration.

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
- [Testing Requirements](#testing-requirements)
- [Open Questions](#open-questions)
- Decision Record
- [References](#references)

---

## Background & Context

### Current State

**wolfharness's History Management:**

AgentPool currently manages conversation history through:

1. **MessageHistory** (`src/wolfharness/messaging/message_history.py`):
   - Stores conversation state in `chat_messages: ChatMessageList`
   - Applies `MemoryConfig` limits (`max_tokens`, `max_messages`)
   - Loads history from storage via `SessionQuery`

2. **CompactionPipeline** (`src/wolfharness/messaging/compaction.py`):
   - Pipeline-based system with predefined steps (FilterThinking, TruncateToolOutputs, KeepLastMessages, etc.)
   - Configurable via YAML with presets (`minimal`, `balanced`, `summarizing`)
   - Applied during message history preparation

3. **NativeAgent** (`src/wolfharness/agents/native_agent/agent.py`):
   - Wraps pydantic-ai's `Agent` class
   - Creates `agentlet` via `get_agentlet()` which instantiates pydantic-ai Agent
   - Currently does NOT pass `history_processors` parameter to pydantic-ai Agent

**pydantic-ai's History Processors:**

pydantic-ai supports `history_processors` as a first-class feature:

```python
# Four supported signatures:
_HistoryProcessorSync = Callable[[list[ModelMessage]], list[ModelMessage]]
_HistoryProcessorAsync = Callable[[list[ModelMessage]], Awaitable[list[ModelMessage]]]
_HistoryProcessorSyncWithCtx = Callable[[RunContext[DepsT], list[ModelMessage]], list[ModelMessage]]
_HistoryProcessorAsyncWithCtx = Callable[[RunContext[DepsT], list[ModelMessage]], Awaitable[list[ModelMessage]]]

HistoryProcessor = (
    _HistoryProcessorSync
    | _HistoryProcessorAsync
    | _HistoryProcessorSyncWithCtx[DepsT]
    | _HistoryProcessorAsyncWithCtx[DepsT]
)
```

Key characteristics:
- Takes a list of `ModelMessage` objects and returns modified list
- Can be sync or async
- Can optionally take `RunContext` to access dependencies, usage stats, model info
- Applied in sequence, with each processor receiving output of previous one
- Executed in `ModelRequestNode._prepare_request()` before sending to model
- Replaces entire message history in state

### Historical Context

- **RFC-0002** established the pattern for extending pydantic-ai features by passing through configuration parameters (e.g., `prepare` hooks, `function_schema`)
- AgentPool's `CompactionPipeline` was designed before pydantic-ai had native history processors, leading to overlapping functionality
- Current wolfharness users must write custom hooks or CompactionSteps to achieve what pydantic-ai's history processors can do natively

### Glossary

| Term | Definition |
|------|------------|
| History Processor | A callable that transforms message history before model invocation (pydantic-ai concept) |
| CompactionPipeline | AgentPool's internal message transformation system |
| MessageHistory | AgentPool's conversation state manager |
| RunContext | pydantic-ai's runtime context object providing access to deps, usage, model info |
| Agentlet | Internal pydantic-ai Agent instance created by wolfharness's `get_agentlet()` |

---

## Problem Statement

### The Problem

AgentPool lacks a native way to configure pydantic-ai's history processors, forcing users to either:

1. **Write custom hooks**: Implement `pre_run` hooks that manipulate `MessageHistory`, which is complex and error-prone
2. **Use CompactionPipeline**: Limited to predefined steps (FilterThinking, TruncateToolOutputs, etc.) with no RunContext access
3. **Cannot access RunContext**: CompactionPipeline and hooks don't provide access to pydantic-ai's RunContext (dependencies, usage stats, model info)
4. **Duplicate functionality**: AgentPool's CompactionPipeline overlaps with pydantic-ai's history processor ecosystem

### Evidence

- **GitHub Issue**: Users request context-aware history management (e.g., "reduce history when token usage exceeds X")
- **pydantic-ai Documentation**: Highlights history processors as the recommended pattern for advanced conversation management
- **RFC-0002 Pattern**: Previous work showed that passing pydantic-ai parameters through configuration is the preferred approach

### Impact of Inaction

- **Cost**: Users cannot implement token-aware history optimization, leading to higher API costs
- **Risk**: Complicated workarounds via hooks may introduce bugs in history manipulation
- **Opportunity**: Missing out on pydantic-ai's growing ecosystem of third-party history processors

---

## Goals & Non-Goals

### Goals (In Scope)

1. Enable YAML configuration of pydantic-ai history processors in `NativeAgentConfig`
2. Support all four history processor signatures (sync/async, with/without RunContext)
3. Allow import path references (string) for processors
4. Pass configured processors to pydantic-ai Agent during agentlet creation
5. Maintain backward compatibility (no processors = existing behavior)
6. Provide type safety and proper error handling for processor configuration
7. Cache resolved processors to avoid repeated import resolution

### Non-Goals (Out of Scope)

1. Replacing or deprecating CompactionPipeline (it remains for existing users)
2. Creating wolfharness-specific history processor abstractions (use pydantic-ai's directly)
3. **Inline code execution for history processors** - deferred to future RFC with security design
4. History processor validation beyond basic import path verification and callable check
5. Runtime debugging of history processor execution (rely on pydantic-ai's built-in logs)

### Success Criteria

- [ ] User can configure history processors via YAML with import paths
- [ ] Processors receive correct RunContext when defined with ctx parameter
- [ ] Async processors execute non-blocking
- [ ] Configuration errors are caught at agent initialization time
- [ ] Existing agents without history processors work unchanged
- [ ] Resolved processors are cached per agent instance

---

## Evaluation Criteria

| Criterion | Weight | Description | Minimum Threshold |
|-----------|--------|-------------|-------------------|
| **Code Simplicity** | High | Minimal changes to existing codebase | < 200 lines new/modified |
| **Type Safety** | High | Compile-time type checking with mypy | Passes mypy --strict |
| **Backward Compatibility** | High | Existing configurations work unchanged | Zero breaking changes |
| **Feature Completeness** | Medium | Supports all pydantic-ai processor signatures | All 4 signatures work |
| **Usability** | Medium | Easy to configure in YAML | Clear documentation, examples |
| **Performance** | Low | No significant performance overhead | < 5% overhead per run |

---

## Options Analysis

### Option 1: Pass-through to pydantic-ai Agent (Recommended)

**Description**

Add a `history_processors` field to `MemoryConfig` that accepts a list of import path strings. The `NativeAgent` resolves these paths to callables and passes them to pydantic-ai's `Agent` constructor during `get_agentlet()`.

**Configuration Schema:**
```yaml
agents:
  my_agent:
    type: native
    model: "openai:gpt-4o"
    session:
      history_processors:
        - "my_module:keep_recent_messages"
        - "my_module:context_aware_filter"
        - "my_module:summarize_old_messages"
```

**Implementation:**
1. Add `history_processors: list[str] | None` field to `MemoryConfig`
2. In `NativeAgent.get_agentlet()`, resolve import paths to callables using existing `import_callable()` from `wolfharness.utils.importing`
3. Cache resolved processors on the agent instance to avoid repeated resolution
4. Pass processors to pydantic-ai Agent's `history_processors` parameter
5. Validate processors are callable at import time

**Advantages**
- **Minimal changes**: ~150 lines total (config field, resolution logic, caching, pass-through)
- **Full pydantic-ai compatibility**: Supports all 4 signatures natively
- **Type safe**: Uses pydantic-ai's type checking
- **No reinvention**: Uses battle-tested pydantic-ai processor execution logic
- **Clean separation**: Processor logic lives in pydantic-ai, not wolfharness
- **Performance**: One-time import resolution per agent instance (cached)
- **Security**: No inline code execution, only import paths

**Disadvantages**
- **Duplication with CompactionPipeline**: Two mechanisms for similar use cases
- **Learning curve**: Users need to understand both CompactionPipeline and history processors

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Code Simplicity | **Excellent** | Only config + pass-through, no new execution logic |
| Type Safety | **Excellent** | pydantic-ai handles types, we just configure |
| Backward Compatibility | **Excellent** | Optional field, defaults to empty list |
| Feature Completeness | **Excellent** | All 4 signatures supported via pydantic-ai |
| Usability | **Good** | YAML config with import paths, familiar pattern |
| Performance | **Excellent** | Cached resolution, zero overhead after init |

**Effort Estimate**
- Complexity: **Low**
- Resources: 1 developer, 2-3 days
- Dependencies: None (purely additive)

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Processor import fails at runtime | Low | Low | Validate imports at agent initialization |
| RunContext injection issues | Low | Low | Rely on pydantic-ai's ctx detection |
| Confusion with CompactionPipeline | Medium | Low | Document that they serve different purposes |

---

### Option 2: Hook-based History Processing

**Description**

Create a new hook event `pre_model_call` that provides access to message history. Users implement hooks to modify history, similar to existing `pre_run` hooks.

**Configuration Schema:**
```yaml
agents:
  my_agent:
    hooks:
      pre_model_call:
        - type: import
          import_path: "my_module:process_history"
```

**Implementation:**
1. Add `pre_model_call` to hook event types
2. Create `HookContext` with message history access
3. In `NativeAgent._stream_events()`, call hooks before `agentlet.iter()`
4. Apply returned history to agentlet's message_history parameter

**Advantages**
- **Familiar pattern**: Hooks are already well-understood by users
- **Flexible**: Can do more than history processing (logging, validation, etc.)
- **Reuses infrastructure**: Hooks already have import path resolution, context injection

**Disadvantages**
- **No RunContext access**: AgentPool's hooks don't provide pydantic-ai's RunContext (usage stats, deps)
- **Async overhead**: Hooks add extra async roundtrip per model call
- **Execution timing**: Runs AFTER agentlet creation, so history passed to pydantic-ai must be modified in-place
- **More complex**: Need to handle both hook and non-hook code paths, sync state management
- **Limited signatures**: Only supports async callables (hook system limitation)

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Code Simplicity | **Poor** | New hook type, timing complexities, state management |
| Type Safety | **Good** | Hooks are typed, but limited to async |
| Backward Compatibility | **Excellent** | Optional, no breaking changes |
| Feature Completeness | **Poor** | Only async, no RunContext access |
| Usability | **Good** | Familiar hook pattern for existing users |
| Performance | **Poor** | Extra async call per model invocation |

**Effort Estimate**
- Complexity: **Medium**
- Resources: 1 developer, 4-5 days
- Dependencies: Hook system architecture

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Timing issues with history application | Medium | High | Careful testing of state synchronization |
| Hook execution blocking | Low | Medium | Enforce timeouts on hooks |

---

### Option 3: CompactionPipeline Enhancement

**Description**

Extend `CompactionPipeline` to support custom steps with RunContext access and pydantic-ai message types, then apply pipeline to history before passing to pydantic-ai Agent.

**Configuration Schema:**
```yaml
agents:
  my_agent:
    session:
      compaction:
        steps:
          - type: custom
            import_path: "my_module:context_aware_step"
            # AgentPool provides a modified context object
```

**Implementation:**
1. Add `CustomCompactionStep` type with RunContext-like object
2. Create adapter layer to convert AgentPool's message types to pydantic-ai's `ModelMessage`
3. Apply CompactionPipeline in `get_agentlet()` before passing history to pydantic-ai
4. Map AgentPool context to pydantic-ai RunContext (partial mapping)

**Advantages**
- **Unified paradigm**: All history manipulation in one place
- **Familiar to existing users**: Extends existing compaction config
- **Centralized logic**: All transformation in CompactionPipeline

**Disadvantages**
- **No true RunContext**: Can only simulate RunContext, not provide pydantic-ai's full context
- **Type conversion overhead**: Must convert between AgentPool and pydantic-ai message types
- **Duplication risk**: Reinventing history processor logic
- **Complex adapter**: Maintaining message type mapping is an ongoing burden
- **Breaking change**: Modifies how CompactionPipeline integrates with Agent

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Code Simplicity | **Poor** | Message type adapters, context mapping, complex |
| Type Safety | **Medium** | Type conversions introduce type gaps |
| Backward Compatibility | **Poor** | May affect existing compaction usage |
| Feature Completeness | **Medium** | Can simulate but not fully support RunContext |
| Usability | **Good** | Familiar compaction pattern |
| Performance | **Medium** | Type conversion overhead per run |

**Effort Estimate**
- Complexity: **High**
- Resources: 1 developer, 5-7 days
- Dependencies: Message conversion infrastructure

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Message type conversion bugs | High | High | Comprehensive test coverage for edge cases |
| RunContext mapping incomplete | Medium | Medium | Document limitations clearly |

---

### Options Comparison Summary

| Criterion | Option 1: Pass-through | Option 2: Hook-based | Option 3: Compaction Enhancement |
|-----------|------------------------|----------------------|-------------------------------|
| Code Simplicity | **Excellent** | Poor | Poor |
| Type Safety | **Excellent** | Good | Medium |
| Backward Compatibility | **Excellent** | Excellent | Poor |
| Feature Completeness | **Excellent** | Poor | Medium |
| Usability | **Good** | Good | Good |
| Performance | **Excellent** | Poor | Medium |
| **Overall** | **Excellent** | Poor | Poor |

---

## Recommendation

### Recommended Option

**Option 1: Pass-through to pydantic-ai Agent**

### Justification

Option 1 scores highest on the most important criteria:

1. **Code Simplicity (High Weight)**: Only ~150 lines of changes vs 200+ for other options. No new execution logic—just configuration and pass-through.

2. **Type Safety (High Weight)**: Leverages pydantic-ai's battle-tested type system. No custom type conversions or adapters needed.

3. **Feature Completeness (Medium Weight)**: Native support for all 4 processor signatures (sync/async, with/without RunContext). Other options have significant limitations here.

4. **Performance (Low Weight)**: Zero additional overhead with one-time import resolution per agent instance. Processors execute exactly as pydantic-ai intended.

5. **Backward Compatibility (High Weight)**: Optional field with sensible default. Zero breaking changes to existing agents.

The pattern also aligns with **RFC-0002**, which established pass-through as the preferred approach for extending pydantic-ai features (e.g., `prepare` hooks, `function_schema`).

### Accepted Trade-offs

1. **Duplication with CompactionPipeline**: Acceptable because they serve different purposes:
   - CompactionPipeline: Pre-configured, declarative transformations (good for simple cases)
   - History processors: Programmatic, context-aware transformations (good for complex cases)
   - Documentation will clarify when to use each and their relationship

2. **Learning curve for users**: Acceptable because pydantic-ai's history processors are well-documented and follow standard callable patterns. The import path syntax is already familiar from tools and hooks.

3. **No inline code execution**: Intentional design choice to avoid security risks. Users who need inline capabilities can use Python files and import paths.

### Conditions

- **Documentation requirement**: Must clearly explain the relationship between CompactionPipeline and history processors with use case guidance
- **Validation**: Must catch import errors at agent initialization, not runtime
- **Execution order**: Must specify that CompactionPipeline (if configured) runs first, then history processors

---

## Technical Design

> Note: This is preliminary design for review. Complete after RFC approval.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│ YAML Configuration                                            │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ agents:                                                     │ │
│ │   my_agent:                                                  │ │
│ │     type: native                                              │ │
│ │     model: "openai:gpt-4o"                              │ │
│ │     session:                                                    │ │
│ │       history_processors: ["module:processor", ...]           │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│ NativeAgent.from_config()                                       │
│   - Parse MemoryConfig.history_processors                        │
│   - Store in agent instance                                     │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│ NativeAgent.get_agentlet()                                      │
│   - Check cache for resolved processors                            │
│   - Resolve import paths to callables (if not cached)        │
│   - Cache resolved processors on instance                       │
│   - Pass to pydantic-ai Agent:                                │
│     history_processors=[processor1, processor2, ...]        │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│ pydantic-ai Agent (no changes)                                 │
│   - ModelRequestNode._prepare_request()                      │
│   - Apply processors sequentially                                  │
│   - Replace state.message_history                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Execution Order (CompactionPipeline vs History Processors)

When both CompactionPipeline and history processors are configured:

1. **CompactionPipeline** (if configured) - Applied by MessageHistory.get_history() BEFORE agentlet creation
2. **History Processors** (pydantic-ai native) - Applied by ModelRequestNode during model request preparation

```python
# In NativeAgent._stream_events():
history_list = message_history.get_history()  # CompactionPipeline applied here
# ...
agentlet = await self.get_agentlet(...)  # History processors configured here
async with agentlet.iter(prompts, message_history=[m for run in history_list for m in run.to_pydantic_ai()]) as agent_run:
    # History processors applied by pydantic-ai internally
```

### Key Components

#### 1. MemoryConfig Extension

**Location**: `src/wolfharness_config/session.py`

```python
class MemoryConfig(Schema):
    """Configuration for agent memory and history handling."""

    # ... existing fields (enable, max_tokens, max_messages, session, provider) ...

    history_processors: list[str] | None = Field(
        default=None,
        examples=[
            ["my_processors:keep_recent_messages", "my_module:summarize_old"]
        ],
        title="History processors",
    )
    """List of import paths to history processor callables.
    
    History processors are applied by pydantic-ai before each model call to
    transform the message history. They can:
    - Filter messages based on content or metadata
    - Truncate or summarize old messages
    - Make context-aware decisions using RunContext (usage, deps, model info)
    
    Each processor must be callable and accept one of these signatures:
    - def processor(messages: list[ModelMessage]) -> list[ModelMessage]
    - async def processor(messages: list[ModelMessage]) -> list[ModelMessage]
    - def processor(ctx: RunContext, messages: list[ModelMessage]) -> list[ModelMessage]
    - async def processor(ctx: RunContext, messages: list[ModelMessage]) -> list[ModelMessage]
    
    See: https://ai.pydantic.dev/history-processors/
    """
```

#### 2. Processor Resolution and Caching

**Location**: `src/wolfharness/agents/native_agent/agent.py`

```python
class Agent[TDeps = None, OutputDataT = str](BaseAgent[TDeps, OutputDataT]):
    """The main agent class."""

    def __init__(self, ...):
        # ... existing __init__ code ...
        self._resolved_history_processors: list[Callable[..., Any]] | None = None  # Cache

    async def get_agentlet[AgentOutputType](
        self,
        model: ModelType | None,
        output_type: type[AgentOutputType] | None,
        input_provider: InputProvider | None,
    ) -> PydanticAgent[AgentContext[TDeps], AgentOutputType]:
        """Create pydantic-ai agent from current state."""
        from wolfharness.utils.importing import import_callable

        # ... existing tool wrapping code ...

        # Resolve history processors (cached)
        processors: list[Callable[..., Any]] = []
        if self._resolved_history_processors is None:
            if self._agent_config.session and self._agent_config.session.history_processors:
                for import_path in self._agent_config.session.history_processors:
                    try:
                        processor = import_callable(import_path)
                        if not callable(processor):
                            raise ValueError(
                                f"History processor import path {import_path!r} "
                                f"does not resolve to a callable"
                            )
                        processors.append(processor)
                    except Exception as e:
                        raise ValueError(
                            f"Failed to import history processor from {import_path!r}: {e}"
                        ) from e
            self._resolved_history_processors = processors
        elif self._resolved_history_processors:
            processors = self._resolved_history_processors

        return PydanticAgent(
            name=self.name,
            model=model_,
            model_settings=self.model_settings,
            instructions=self._formatted_system_prompt,
            retries=self._retries,
            end_strategy=self._end_strategy,
            output_retries=self._output_retries,
            deps_type=AgentContext[TDeps],
            output_type=cast(Any, final_type),
            tools=pydantic_ai_tools,
            builtin_tools=self._builtin_tools,
            history_processors=processors,  # Pass through to pydantic-ai
        )
```

### Data Model

```
MemoryConfig (extended)
├── enable: bool
├── max_tokens: int | None
├── max_messages: int | None
├── session: SessionQuery | None
├── provider: str | None
└── history_processors: list[str] | None  # NEW: Import paths to callables
```

### API Design

**No new API surface**—purely configuration-driven.

**Configuration Examples:**

```yaml
# Example 1: Simple import-based processor
agents:
  coder:
    type: native
    model: "openai:gpt-4o"
    session:
      history_processors:
        - "my_processors:keep_recent_messages"

# Example 2: Context-aware processor
agents:
  coder:
    type: native
    model: "openai:gpt-4o"
    session:
      history_processors:
        - "my_processors:token_aware_filter"
        # Processor signature: def processor(ctx: RunContext, messages: list[ModelMessage]) -> list[ModelMessage]

# Example 3: Multiple processors (pipeline)
agents:
  coder:
    type: native
    model: "openai:gpt-4o"
    session:
      history_processors:
        - "my_processors:filter_thinking"
        - "my_processors:token_budget_keeper"
        - "my_processors:summarize_old_messages"

# Example 4: Combined with CompactionPipeline
agents:
  coder:
    type: native
    model: "openai:gpt-4o"
    session:
      compaction:
        steps:
          - type: filter_thinking  # Runs first
      history_processors:
        - "my_processors:token_aware_filter"  # Runs second (in pydantic-ai)
```

---

## Security Considerations

### Threat Analysis

| Threat | Impact | Likelihood | Mitigation |
|--------|--------|------------|------------|
| Malicious processor import | High | Low | Users control their own imports; document security responsibilities |
| History manipulation via processors | Medium | Low | This is the intended behavior of history processors |
| Denial-of-service via infinite loop | Medium | Low | pydantic-ai handles processor timeouts internally |
| Information leakage via processor | Medium | Low | Processor runs on data already in agent memory |

### Security Measures

- [x] No inline code execution (only import paths from v1)
- [ ] Validate import paths at agent initialization (fails fast, not runtime)
- [ ] Document security implications of history processors clearly
- [ ] Add unit tests for processor validation

### Compliance

No regulatory implications identified. History processors operate on data already in memory.

---

## Implementation Plan

### Phases

#### Phase 1: Configuration Models (0.5 day)

**Scope**: Add configuration types for history processors

**Deliverables**:
- `history_processors: list[str] | None` field added to `MemoryConfig`
- Documentation in field docstring

**Dependencies**: None

#### Phase 2: Resolution Logic & Caching (1 day)

**Scope**: Implement processor import resolution with caching

**Deliverables**:
- `_resolved_history_processors` instance attribute
- Resolution logic in `get_agentlet()` using `import_callable()`
- Callable validation and error handling

**Dependencies**: Phase 1 complete

#### Phase 3: Integration (0.5 day)

**Scope**: Pass processors to pydantic-ai Agent

**Deliverables**:
- Pass `history_processors=processors` to PydanticAgent constructor
- Integration testing

**Dependencies**: Phase 2 complete

#### Phase 4: Testing & Documentation (2 days)

**Scope**: Test coverage and user documentation

**Deliverables**:
- Unit tests for config validation
- Unit tests for resolution logic
- Integration tests with actual pydantic-ai agents (using TestModel)
- Documentation with examples
- Migration guide: CompactionPipeline vs history processors

**Dependencies**: Phase 3 complete

### Milestones

| Milestone | Description | Target | Status |
|-----------|-------------|--------|--------|
| Config models defined | history_processors field added to MemoryConfig | Day 1 | Not Started |
| Resolution complete | Processors resolve from config to callables with caching | Day 2 | Not Started |
| Integration working | Processors passed to pydantic-ai Agent | Day 3 | Not Started |
| Tests passing | All tests green, coverage adequate | Day 5 | Not Started |
| Documentation | User-facing docs published | Day 5 | Not Started |

### Rollback Strategy

If issues arise:
1. Revert `history_processors` field addition to `MemoryConfig`
2. Remove `_resolved_history_processors` attribute and resolution logic
3. Revert `get_agentlet()` to not pass `history_processors` parameter
4. Delete any added test files

The rollback is straightforward because changes are purely additive.

---

## Testing Requirements

### Unit Tests (`wolfharness/tests/test_history_processors.py`)

**Configuration Validation**
- [ ] Empty history_processors list is valid
- [ ] history_processors=None is valid (default)
- [ ] Invalid import path raises ValueError with clear message
- [ ] Import path that's not callable raises ValueError

**Processor Resolution**
- [ ] Sync callable imported correctly
- [ ] Async callable imported correctly
- [ ] Context-aware callable (with RunContext) imported
- [ ] Multiple processors all imported
- [ ] Import errors surface with user-friendly messages
- [ ] Resolved processors are cached (only called once per agent)

### Integration Tests (with TestModel)

**Processor Behavior**
- [ ] Processor receives correct message history
- [ ] Processor return value replaces history
- [ ] RunContext injection works (usage, deps available)
- [ ] Multiple processors execute in sequence (output of processor N is input to N+1)
- [ ] Processor exceptions propagate correctly

**Compatibility**
- [ ] Agents without history_processors work unchanged
- [ ] CompactionPipeline + history_processors work together correctly

### Regression Tests

- [ ] Existing agent tests pass without changes
- [ ] MemoryConfig serialization/deserialization works
- [ ] Session loading doesn't break

### Test Utilities

Create example processors for testing:

```python
# test_processors.py
from pydantic_ai import RunContext, ModelRequest, ModelResponse

def keep_recent_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Keep only last 5 messages (simple sync)."""
    return messages[-5:] if len(messages) > 5 else messages

async def filter_thinking_async(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Remove thinking parts (simple async)."""
    filtered: list[ModelMessage] = []
    for msg in messages:
        if isinstance(msg, ModelResponse):
            has_content = any(p for p in msg.parts if not p.is_thinking())
            if has_content:
                filtered.append(msg)
        else:
            filtered.append(msg)
    return filtered

def context_aware_sync(
    ctx: RunContext[None],
    messages: list[ModelMessage],
) -> list[ModelMessage]:
    """Reduce history based on token usage (context-aware sync)."""
    if ctx.usage.total_tokens > 5000:
        return messages[-3:]
    return messages
```

---

## Open Questions

1. **None at this time** - Metis review addressed all critical questions.

---

## Decision Record

> Complete this section after RFC review is concluded.

### Decision

**Status**: [APPROVED / REJECTED / DEFERRED]

**Date**: YYYY-MM-DD

**Approvers**
- [Name 1]
- [Name 2]

### Decision Summary

[Brief statement of decision made]

### Key Discussion Points

[Notable points raised during review that influenced decision]

1. [Point 1]
2. [Point 2]

### Conditions of Approval

[Any conditions or modifications required]

### Dissenting Opinions

[Document any significant disagreements for the record]

---

## References

### Related Documents

- [RFC-0002: Extended Tool Definition](./RFC-0002-extended-tool-definition.md)
- pydantic-ai Documentation - History Processors
- pydantic-ai Test Suite

### External Resources

- [pydantic-ai History Processors Guide](https://ai.pydantic.dev/history-processors/)
- [pydantic-ai RunContext API](https://ai.pydantic.dev/run-context/)
- [AgentPool Configuration Docs](https://million-mo.github.io/wolfharness/YAML%20Configuration/session_configuration/)

### Appendix

#### pydantic-ai History Processor Signatures (Reference)

```python
# Type 1: Simple sync processor
def simple_processor(messages: list[ModelMessage]) -> list[ModelMessage]:
    return messages[-10:]

# Type 2: Simple async processor
async def async_processor(messages: list[ModelMessage]) -> list[ModelMessage]:
    return messages[-10:]

# Type 3: Context-aware sync processor
def context_sync_processor(
    ctx: RunContext[None],
    messages: list[ModelMessage],
) -> list[ModelMessage]:
    if ctx.usage.total_tokens > 5000:
        return messages[-5:]
    return messages

# Type 4: Context-aware async processor
async def context_async_processor(
    ctx: RunContext[MyDeps],
    messages: list[ModelMessage],
) -> list[ModelMessage]:
    # Can access ctx.deps for dependencies
    # Can access ctx.usage for token stats
    # Can access ctx.model for model info
    if ctx.usage.total_tokens > 10000:
        return [messages[-1]]  # Keep only current prompt
    return messages
```

#### Example History Processor Implementations

```python
# my_processors.py
from pydantic_ai import RunContext, ModelRequest, ModelResponse

def keep_recent_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Keep only last 10 messages."""
    return messages[-10:] if len(messages) > 10 else messages

def filter_thinking(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Remove ModelResponse messages that only contain thinking parts."""
    filtered: list[ModelMessage] = []
    for msg in messages:
        if isinstance(msg, ModelResponse):
            # Check if message has non-thinking parts
            has_content = any(p for p in msg.parts if not p.is_thinking())
            if has_content:
                filtered.append(msg)
        else:
            filtered.append(msg)
    return filtered

def token_aware_filter(
    ctx: RunContext[None],
    messages: list[ModelMessage],
) -> list[ModelMessage]:
    """Reduce history when token usage is high."""
    # Dynamic threshold based on current usage
    if ctx.usage.total_tokens > 8000:
        return messages[-3:]  # Aggressive reduction
    elif ctx.usage.total_tokens > 5000:
        return messages[-7:]  # Moderate reduction
    return messages

async def summarize_old_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Summarize old messages when conversation is long."""
    if len(messages) > 20:
        # First 10 messages to summarize
        old_messages = messages[:10]
        # Last 10 messages (keep as-is)
        recent = messages[-10:]

        # Use a summarizer agent
        from pydantic_ai import Agent
        summarizer = Agent('openai:gpt-4o-mini', instructions="Summarize...")
        summary_result = await summarizer.run(message_history=old_messages)

        # Return summary + recent messages
        return summary_result.all_messages() + recent
    return messages
```

#### Migration Guide: CompactionPipeline to History Processors

| CompactionPipeline Step | History Processor Equivalent |
|-------------------------|------------------------------|
| FilterThinking() | `filter_thinking(messages)` |
| KeepLastMessages(10) | `lambda msgs: msgs[-10:]` |
| TruncateToolOutputs(1000) | Custom processor to truncate content |
| SummarizeOld(model="gpt-4") | `summarize_old_messages(ctx, msgs)` |

**When to use CompactionPipeline:**
- Simple, declarative transformations
- No need for RunContext access
- Prefer YAML-only configuration

**When to use History Processors:**
- Context-aware logic (based on token usage, dependencies)
- Complex transformations (summarization, semantic filtering)
- Need full control over message manipulation
