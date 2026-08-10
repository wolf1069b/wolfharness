---
rfc_id: RFC-0017
title: OpenCode Command Endpoint Skill Support
description: Modified the /session/{id}/command endpoint in the OpenCode server to execute slashed commands (including skill commands) in addition to MCP prompts, resolving the issue where skill commands return 404.
author: OpenCode Team
reviewers: []
created: 2025-03-18
last_updated: 2025-03-18
decision_date: null
---

# RFC-0017: OpenCode Command Endpoint Skill Support

## Overview

The current OpenCode server `/session/{id}/command` endpoint only executes MCP Prompts, causing skill commands exposed as slashed Commands to return 404. This RFC proposes modifying the existing endpoint to support both MCP Prompts and slashed Commands, enabling skills to be invoked via the native `/skill:name` command syntax.

**Current Behavior**: `POST /session/{id}/command` with `skill:test-skill` returns 404 Not Found.

**Proposed Behavior**: Same request executes the skill command and loads instructions into agent context.

## Problem Statement

### Current Implementation (src/wolfharness_server/opencode_server/routes/session_routes.py)

```python
@router.post("/{session_id}/command")
async def execute_command(...):
    """Execute a slash command (MCP prompt)."""
    prompts = await state.agent.tools.list_prompts()
    prompt = next((p for p in prompts if p.name == request.command), None)
    if prompt is None:
        raise HTTPException(status_code=404, detail="Command not found")
    # ... execute MCP prompt
```

**Issue**: The endpoint only searches `list_prompts()`. Slashed Commands (from `CommandStore`) are never checked.

### Impact

| Feature | Expected | Actual |
|---------|----------|--------|
| MCP Prompts in `/command` | Works | ✅ Works |
| Slashed Commands in `/command` | Works | ❌ Returns 404 |
| Skill Commands via `/skill:name` | Loads skill | ❌ Command not found |

### Root Cause

**Two Different Command Systems in OpenCode**:

1. **MCP Prompts**: Read-only templates, returned by `GET /command`, executed via POST using `prompt.get_components()`
2. **Slashed Commands**: Executable commands from `CommandStore`, executed via `command.execute(ctx, args)`

The POST endpoint was designed only for MCP Prompts, ignoring slashed Commands.

## Goals

1. Enable skill commands (and other slashed Commands) to be executed via `/session/{id}/command`
2. Maintain backward compatibility with existing MCP Prompt execution
3. Define clear precedence when command names conflict between systems
4. Minimal code changes to existing routing infrastructure

## Non-Goals

1. **NO** adding a new endpoint (e.g., `/session/{id}/slash`) - use existing endpoint
2. **NO** breaking changes to MCP Prompt execution flow
3. **NO** changing the `CommandRequest` or response schema
4. **NO** modifying how `GET /command` aggregates commands (it already includes both)

## Evaluation Criteria

| Criterion | Weight | Description | Min Threshold |
|-----------|--------|-------------|---------------|
| Backward Compatibility | Critical | Existing MCP prompts continue to work | 100% compatible |
| Performance | High | Command lookup <5ms overhead | <10ms acceptable |
| Code Simplicity | High | Minimal changes to session_routes.py | <50 LOC modified |
| Precedence Clarity | Medium | Clear rules for command resolution | Documented behavior |

## Options Analysis

### Option A: Extend Existing Endpoint (Recommended)

**Description**: Modify `execute_command()` to check `CommandStore` before falling back to MCP Prompts.

```python
@router.post("/{session_id}/command")
async def execute_command(...):
    # 1. Check slashed Commands first (skills, etc.)
    if state.command_store and request.command in state.command_store:
        return await _execute_slashed_command(state, request)
    
    # 2. Fall back to MCP Prompts
    prompts = await state.agent.tools.list_prompts()
    prompt = next((p for p in prompts if p.name == request.command), None)
    if prompt is None:
        raise HTTPException(status_code=404, detail="Command not found")
    # ... existing MCP prompt execution
```

**Advantages**:
- Single endpoint, simpler API contract
- No client changes required
- Minimal code modifications
- Follows existing pattern (GET /command already aggregates both)

**Disadvantages**:
- Two different execution paths in one function
- Need to handle different return types gracefully
- Precedence rules must be documented

**Evaluation Against Criteria**:
| Criterion | Score | Notes |
|-----------|-------|-------|
| Backward Compatibility | ✅ A+ | Existing prompts continue to work |
| Performance | ✅ A | Single additional dict lookup |
| Code Simplicity | ✅ A | ~30 LOC added |
| Precedence Clarity | ✅ B | Need documentation |

**Effort Estimate**: 2-3 days (including tests)

---

### Option B: New Dedicated Endpoint

**Description**: Create `POST /session/{id}/slash` specifically for slashed Commands.

```python
@router.post("/{session_id}/slash")
async def execute_slash_command(...):
    """Execute slashed command only."""
    if not state.command_store or request.command not in state.command_store:
        raise HTTPException(status_code=404, detail="Slash command not found")
    return await _execute_slashed_command(state, request)
```

**Advantages**:
- Clean separation of concerns
- No precedence ambiguity
- Easier to maintain distinct execution paths

**Disadvantages**:
- New API endpoint to document and maintain
- Clients need to know which endpoint to call
- Inconsistent with `GET /command` (which aggregates both)
- Breaking change for skill command discovery

**Evaluation Against Criteria**:
| Criterion | Score | Notes |
|-----------|-------|-------|
| Backward Compatibility | ❌ C | New endpoint required |
| Performance | ✅ A | Same as Option A |
| Code Simplicity | ❌ C | New file, routing, tests |
| Precedence Clarity | ✅ A | No ambiguity |

**Effort Estimate**: 4-5 days (new endpoint + documentation + client updates)

---

### Option C: Unified Command Abstraction

**Description**: Create a `UnifiedCommandExecutor` that abstracts both MCP Prompts and slashed Commands.

```python
class UnifiedCommandExecutor:
    async def execute(self, command_name: str, arguments: str):
        # Try slashed commands first
        if self._in_command_store(command_name):
            return await self._execute_slashed(command_name, arguments)
        # Fall back to prompts
        if self._in_prompt_store(command_name):
            return await self._execute_prompt(command_name, arguments)
        raise CommandNotFound()
```

**Advantages**:
- Clean abstraction layer
- Reusable across different contexts
- Easier to test

**Disadvantages**:
- More complex initial implementation
- Over-engineering for this specific case
- Delays skill command support

**Evaluation Against Criteria**:
| Criterion | Score | Notes |
|-----------|-------|-------|
| Backward Compatibility | ✅ A | Existing code works |
| Performance | ✅ B | Additional abstraction layer |
| Code Simplicity | ❌ D | Too much abstraction |
| Precedence Clarity | ✅ A | Centralized logic |

**Effort Estimate**: 1-2 weeks (design + implementation + refactoring)

## Recommendation

**Recommended Option**: **Option A** - Extend Existing Endpoint

**Justification**:
- Meets all critical criteria (backward compatibility, performance)
- Minimal code changes reduce risk
- Consistent with existing `GET /command` behavior (which aggregates both)
- Fastest time-to-value for skill command support

**Acknowledged Trade-offs**:
- Execution logic will have two paths (acceptable complexity)
- Precedence rules must be documented clearly

## Technical Design

### Execution Precedence

**Command Resolution Order**:
1. Check `CommandStore` for slashed Commands (skills, custom commands)
2. If not found, check MCP Prompts
3. If neither found, return 404

**Rationale**: Skills should take precedence over generic prompts with same name.

### Implementation Details

**Modified File**: `src/wolfharness_server/opencode_server/routes/session_routes.py`

**New Helper Function**:

```python
async def _execute_slashed_command(
    state: FastAPIState,
    request: CommandRequest,
) -> MessageWithParts:
    """Execute slashed command and return result."""
    if not state.command_store:
        raise HTTPException(status_code=500, detail="Command store not initialized")
    
    command = state.command_store.get_command(request.command)
    if command is None:
        raise HTTPException(status_code=404, detail="Command not found")
    
    # Create command context
    ctx = CommandContext(
        agent=state.agent,
        output=CommandOutput(),
        working_dir=state.working_dir,
    )
    
    # Parse arguments
    args = request.arguments.split() if request.arguments else []
    
    # Execute command
    try:
        result = await command.execute(ctx, args)
        
        # Create response message
        return MessageWithParts(
            role="assistant",
            parts=[TextPart(type="text", text=str(result) if result else "Command executed")],
            model=request.model,
            provider="opencode",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Command failed: {e}")
```

**Modified Endpoint**:

```python
@router.post("/{session_id}/command")
async def execute_command(
    session_id: str,
    request: CommandRequest,
    state: StateDep,
) -> MessageWithParts:
    """Execute a slash command (MCP prompt or slashed command).
    
    Commands are resolved in order:
    1. Slashed commands from CommandStore (skills, etc.)
    2. MCP prompts from list_prompts()
    3. Return 404 if neither found
    """
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # 1. Try slashed commands first (skills take precedence)
    if state.command_store and request.command in state.command_store:
        return await _execute_slashed_command(state, request)
    
    # 2. Fall back to MCP prompts (original behavior)
    prompts = await state.agent.tools.list_prompts()
    prompt = next((p for p in prompts if p.name == request.command), None)
    if prompt is None:
        detail = f"Command not found: {request.command}"
        raise HTTPException(status_code=404, detail=detail)
    
    # ... existing MCP prompt execution (unchanged)
```

### State Management

**CommandStore Integration**:

```python
# In src/wolfharness_server/opencode_server/server.py

async def _setup_skill_commands(self) -> None:
    """Setup command store with skills."""
    if not self._pool.skill_commands.has_skills:
        return
    
    from slashed import CommandStore
    
    command_store = CommandStore()
    for cmd in self._pool.skill_commands.get_commands():
        command_store.register_command(cmd)
    
    self._command_store = command_store

# Make available in state
class FastAPIState:
    command_store: CommandStore | None = None
```

### Error Handling

| Scenario | Status Code | Message |
|----------|-------------|---------|
| Session not found | 404 | "Session not found" |
| Command not in either system | 404 | "Command not found: {name}" |
| Command execution failed | 500 | "Command failed: {error}" |
| Command store not initialized | 500 | "Command store not initialized" |

### Testing Strategy

**Unit Tests**:
```python
# Test slashed command execution
async def test_execute_slash_command():
    state.command_store = MockCommandStore()
    state.command_store.register_command(MockCommand("test"))
    
    result = await execute_command("session-1", CommandRequest(command="test"), state)
    assert result.role == "assistant"

# Test slashed takes precedence over prompt
def test_slash_takes_precedence():
    # Both prompt and command named "test"
    state.command_store.register_command(MockCommand("test"))
    state.agent.tools.list_prompts.return_value = [MockPrompt("test")]
    
    # Should execute command, not prompt
    ...

# Test fallback to prompt
def test_fallback_to_prompt():
    state.command_store = None
    state.agent.tools.list_prompts.return_value = [MockPrompt("test")]
    
    # Should execute prompt
    ...
```

**Integration Tests**:
```bash
# Test skill command execution
curl -X POST http://localhost:8000/session/test-session/command \
  -H "Content-Type: application/json" \
  -d '{"command": "skill:test", "arguments": "arg1 arg2"}'

# Verify: Returns 200 with command output, not 404
```

## Implementation Plan

### Phase 1: Command Store Integration (Day 1)

1. Add `CommandStore` to `FastAPIState` and server initialization
2. Create `_setup_skill_commands()` in server.py
3. Unit test: CommandStore initialization

**Deliverable**: CommandStore accessible in endpoint handlers

### Phase 2: Slashed Command Execution (Day 2)

1. Implement `_execute_slashed_command()` helper
2. Modify `execute_command()` endpoint to check CommandStore first
3. Add precedence logic (slashed > prompt)

**Deliverable**: Both command types executable

### Phase 3: Testing & Validation (Day 3)

1. Unit tests for both execution paths
2. Integration tests with mock skills
3. Verify backward compatibility with existing MCP prompts
4. Performance benchmark (<5ms overhead)

**Deliverable**: Full test coverage

### Phase 4: Documentation (Day 4)

1. Update ENDPOINTS.md with new behavior
2. Document precedence rules
3. Add example: executing skill commands
4. Update API changelog

**Deliverable**: Documentation complete

**Total Timeline**: 4 days

## Backward Compatibility

| Scenario | Before | After | Compatible? |
|----------|--------|-------|-------------|
| MCP prompt execution | Works | Works | ✅ Yes |
| New slashed command | 404 | Works | ✅ New feature |
| Name conflict (prompt wins) | Prompt used | Command used | ⚠️ Behavior change |

**Behavior Change Warning**:
If a slashed Command and MCP Prompt have the same name, the slashed Command will now take precedence. This was previously impossible (slashed Commands couldn't be executed), so no existing functionality is broken.

**Mitigation**: Log a warning when both exist:
```python
if state.command_store and request.command in state.command_store:
    # Check if prompt also exists
    prompts = await state.agent.tools.list_prompts()
    if any(p.name == request.command for p in prompts):
        logger.warning(
            "Both slashed command and prompt exist for '{name}'. "
            "Using slashed command.",
            name=request.command
        )
    return await _execute_slashed_command(state, request)
```

## Decision Record

**Status**: REVIEW

**Decision**: Option A - Extend existing /session/{id}/command endpoint to support both slashed Commands and MCP Prompts.

**Conditions for Approval**:
1. Unit tests pass (>80% coverage)
2. Integration tests pass
3. Backward compatibility verified (existing prompts work)
4. Performance benchmark <10ms overhead
5. Documentation updated

**Open Questions**:
1. Should we add metric/logging for which execution path is used?
2. Should the precedence be configurable per-command?

## Appendix A: Current vs Proposed Execution Flow

### Current Flow
```
POST /session/{id}/command
  ↓
list_prompts()
  ↓
Find matching prompt
  ↓
Found? → Execute prompt → Return message
Not found? → 404 Command not found
```

### Proposed Flow
```
POST /session/{id}/command
  ↓
Check CommandStore first
  ↓
Found? → Execute slashed command → Return message
Not found? → list_prompts()
              ↓
              Find matching prompt
                ↓
                Found? → Execute prompt → Return message
                Not found? → 404 Command not found
```

## Appendix B: API Contract

### Request Schema (unchanged)
```json
{
  "command": "skill:test-skill",
  "arguments": "arg1 arg2",
  "model": "optional-model",
  "agent": "optional-agent"
}
```

### Response Schema (unchanged)
```json
{
  "role": "assistant",
  "parts": [{"type": "text", "text": "Command output"}],
  "model": "model-name",
  "provider": "opencode"
}
```

### Error Responses
| Code | Scenario | Body |
|------|----------|------|
| 404 | Session not found | `{"detail": "Session not found"}` |
| 404 | Command not found | `{"detail": "Command not found: {name}"}` |
| 500 | Execution failed | `{"detail": "Command failed: {error}"}` |
