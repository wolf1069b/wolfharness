# SessionPool Global Permission/Question Listing API Design

> **Status**: Accepted — this design has been implemented in the AgentPool codebase.
> See `docs/explanation/` for the current architecture documentation.

## Overview

This document specifies the API for listing pending permissions and questions across all sessions. It replaces `ServerState.pending_questions` as the global source of truth for permission/question state.

## API Surface

**Both message-history (B0.1) and permission/question APIs use `SessionPool` as the canonical API surface.** `SessionPool` delegates to `SessionController` internally. This aligns with the existing pattern where protocol handlers call `SessionPool.receive_request()` and `SessionPool.run_stream()`.

## Background

In Migration A:
- `OpenCodeInputProvider` stores pending questions internally (per-session) — **NOTE: A1.7 must add `_pending_questions` to `OpenCodeInputProvider` first; questions currently live on `ServerState.pending_questions`**
- `ServerState.pending_questions` still exists as a global dict
- A5.8 adds `SessionController.cancel_all_pending_questions()` for SSE disconnect (called via `state.session_controller`)

In Migration B:
- `ServerState.pending_questions` is removed (B4.8)
- All global listing must query `SessionPool`

## API Specification

### Core Types

```python
# wolfharness/models/pending_interaction.py
from typing import Protocol
from datetime import datetime

class PendingQuestion(Protocol):
    """Protocol for pending questions."""
    id: str
    session_id: str
    tool_name: str
    content: str
    created_at: datetime

class PendingPermission(Protocol):
    """Protocol for pending permissions."""
    id: str
    session_id: str
    tool_name: str
    content: str
    created_at: datetime

# wolfharness_server/opencode_server/models/question_permission.py
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class OpenCodePendingQuestion:
    """Concrete implementation for OpenCode protocol."""
    id: str
    session_id: str
    tool_name: str
    content: str
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class OpenCodePendingPermission:
    """Concrete implementation for OpenCode protocol."""
    id: str
    session_id: str
    tool_name: str
    content: str
    created_at: datetime = field(default_factory=datetime.utcnow)
```

### SessionPool API

```python
class SessionPool:
    async def list_pending_questions(
        self,
        *,
        session_id: str | None = None,
        tool_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[PendingQuestion]:
        """List pending questions across all sessions.

        Args:
            session_id: Filter by session ID. If None, returns questions
                from all sessions.
            tool_name: Filter by tool name. If None, returns questions
                from all tools.
            limit: Maximum number of questions to return (default 100).
            offset: Number of questions to skip (default 0).

        Returns:
            List of pending questions ordered by created_at (newest first).
        """

    async def list_pending_permissions(
        self,
        *,
        session_id: str | None = None,
        tool_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[PendingPermission]:
        """List pending permissions across all sessions.

        Args:
            session_id: Filter by session ID. If None, returns permissions
                from all sessions.
            tool_name: Filter by tool name. If None, returns permissions
                from all tools.
            limit: Maximum number of permissions to return (default 100).
            offset: Number of permissions to skip (default 0).

        Returns:
            List of pending permissions ordered by created_at (newest first).
        """

    async def cancel_all_pending_questions(
        self,
        session_id: str | None = None,
    ) -> list[str]:
        """Cancel all pending questions.

        Args:
            session_id: If set, only cancel questions for this session.
                If None, cancel questions across all sessions.

        Returns:
            List of cancelled question IDs.
        """

    async def cancel_all_pending_permissions(
        self,
        session_id: str | None = None,
    ) -> list[str]:
        """Cancel all pending permissions.

        Args:
            session_id: If set, only cancel permissions for this session.
                If None, cancel permissions across all sessions.

        Returns:
            List of cancelled permission IDs.
        """
```

## Implementation Strategy

### Data Collection — Two-Phase Approach

**Phase 1 (Migration A)**: Questions are stored on `ServerState.pending_questions` (global dict), NOT on `input_provider`. `OpenCodeInputProvider` stores permissions internally (`_pending_permissions`), but questions are stored on `ServerState`.

```python
# During Migration A, SessionPool.list_pending_questions() must:
# 1. Iterate ServerState.pending_questions (global dict)
# 2. Also check session.input_provider for permissions

class SessionPool:
    async def list_pending_questions(self, *, session_id=None, tool_name=None, limit=100, offset=0):
        all_questions: list[PendingQuestion] = []
        
        # Source 1: ServerState.pending_questions (Migration A compat)
        # This requires ServerState shim to expose pending_questions
        # or SessionPool maintains its own index
        
        # Source 2: session.input_provider (after A1.7 migration)
        sessions = [self.sessions.get_session(session_id)] if session_id else self.sessions._sessions.values()
        for session in sessions:
            if session is not None and session.input_provider is not None:
                # input_provider has get_pending_questions() after A1.7
                questions = session.input_provider.get_pending_questions()
                for q in questions:
                    if tool_name is None or q.tool_name == tool_name:
                        all_questions.append(q)
        
        # Sort by created_at descending
        all_questions.sort(key=lambda q: q.created_at, reverse=True)
        
        # Apply pagination
        return all_questions[offset:offset + limit]
```

**Phase 2 (Migration B)**: After `ServerState.pending_questions` is removed (B4.8), all questions come from `session.input_provider`.

### Important Correction

The original design incorrectly stated that `OpenCodeInputProvider` stores pending questions. **This is wrong.** Looking at the actual code:

- `input_provider.py:337` and `input_provider.py:514`: Questions are stored via `self._pending_questions_dict[question_id] = PendingQuestion(...)` — this property returns `session.pending_questions` when available, falling back to `self.state.pending_questions`
- `OpenCodeInputProvider` has `_pending_permissions` for permissions, but NO `_pending_questions`.

**Migration A requirement**: A1.7 must add `get_pending_questions()` and `cancel_pending_questions()` to `OpenCodeInputProvider`, migrating question storage from `ServerState` to the provider. Until then, `SessionPool.list_pending_questions()` must read from `ServerState`.

### Revised A1.7 Task

Update A1.7 to explicitly:
1. Add `_pending_questions: dict[str, PendingQuestion]` to `OpenCodeInputProvider`
2. Add `get_pending_questions() -> list[PendingQuestion]` method
3. Add `cancel_pending_questions() -> list[str]` method
4. Migrate question creation from `self.state.pending_questions[id] = ...` to `self._pending_questions[id] = ...`
5. Maintain backward compatibility: if `self._pending_questions` is empty, fall back to `self.state.pending_questions` during Migration A

### Pagination Strategy

**Cursor-based pagination** is preferred over offset-based for real-time data:

```python
# Cursor-based (recommended for live data)
async def list_pending_questions(
    self,
    *,
    session_id: str | None = None,
    tool_name: str | None = None,
    limit: int = 100,
    after_cursor: str | None = None,  # Question ID to start after
) -> tuple[list[PendingQuestion], str | None]:
    """Returns (questions, next_cursor). next_cursor is None when no more results."""
```

However, for simplicity and compatibility with existing HTTP endpoints, **offset-based** is used initially. Cursor-based can be added later without breaking changes.

### Performance Considerations

| Scenario | Complexity | Mitigation |
|----------|-----------|------------|
| 100 sessions, 1 question each | O(100) | Acceptable |
| 1000 sessions, scan all | O(1000) | Cache question counts per session |
| Frequent polling from UI | O(sessions × questions) | Add caching layer with TTL |

**Recommended**: Add an index cache:

```python
class SessionController:
    def __init__(self):
        self._pending_question_index: dict[str, set[str]] = {}  # session_id -> {question_ids}
        self._pending_permission_index: dict[str, set[str]] = {}  # session_id -> {permission_ids}
```

The index is updated when questions/permissions are added or removed, making `list_*` O(1) for the index lookup + O(limit) for result construction.

## Route Updates

### Global Question Listing Endpoint

Current (`question_routes.py`):
```python
@router.get("/")
async def list_questions(state: StateDep):
    pending = _get_all_pending_questions(state)
    return [
        QuestionRequest(
            id=question_id,
            session_id=i.session_id,
            questions=i.questions,
            tool=i.tool,
        )
        for question_id, i in pending.items()
    ]
```

New:
```python
@router.get("/questions")
async def list_questions(session_pool: SessionPool):
    questions = await session_pool.list_pending_questions()
    return [dataclasses.asdict(q) for q in questions]
```

### SSE Disconnect Handler

Current (`global_routes.py:295`):
```python
state.cancel_all_pending_questions()
```

New (from A5.8):
```python
session_controller.cancel_all_pending_questions()
```

## Migration Path

1. **Migration A (A5.4-A5.8)**: Add stub and implementation to `SessionController`
2. **Migration B (B4.8)**: Remove `ServerState.pending_questions`, update routes to use `SessionPool`

## Open Questions

1. **Should we support WebSocket push for permission changes?**
   - Currently the UI polls `/questions`. Real-time push via EventBus would be more efficient.
   - *Decision deferred*: Can be added post-Migration B without API changes.

2. **Permission expiration?**
   - Should pending permissions auto-expire after N minutes?
   - *Recommendation*: Yes, add `expires_at` field and a background cleanup task.
