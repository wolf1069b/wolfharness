# SessionPool Message History API Design

> **Status**: Accepted — this design has been implemented in the AgentPool codebase.
> See `docs/explanation/` for the current architecture documentation.

## Overview

This document specifies **new message history API methods to add to SessionPool**. These methods do not exist yet — they must be implemented as part of Migration B. The API replaces `ServerState.messages` as the canonical message store. All message CRUD operations in the OpenCode Server must route through SessionPool instead of accessing in-memory dicts directly.

## Design Principles

1. **SessionPool is the API surface** for message history — routes call SessionPool methods
2. **StorageProvider is the persistence layer** — accessed via AgentPool, NOT directly by SessionPool
3. **Async-first** — all operations are async to support SQL/remote storage
4. **Backward compatible** — existing StorageProvider implementations continue to work

## Architecture

```
┌─────────────────────────────────────────┐
│         OpenCode Server Routes          │
│   (share_session, revert_session, etc.) │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│           SessionPool API               │
│   get_messages / append_message / ...   │
│                                         │
│   SessionPool has:                      │
│   ├─ self.sessions: SessionController  │
│   │   └─ store: SessionStore           │
│   └─ self.pool: AgentPool              │
│       └─ storage: StorageManager       │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         StorageManager Layer            │
│   Forwards to StorageProvider           │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         StorageProvider Layer           │
│   get_session_messages / log_message    │
│   fork_conversation (existing)          │
└─────────────────────────────────────────┘
```

**Key point**: `SessionPool` lives in `src/wolfharness/orchestrator/core.py` and does not have message history methods yet. It has `self.sessions` (SessionController) and `self.sessions.store` (SessionStore for session metadata). Message history operations must be added to SessionPool as new methods. The storage layer is accessed via `AgentPool.storage`, which is a **`StorageManager`** (not `StorageProvider`). `StorageManager` is a proxy that forwards to configured `StorageProvider` instances.

### Type Conversion

OpenCode Server stores `MessageWithParts` (OpenCode-specific model with `info` + `parts` fields). `StorageProvider` works with `ChatMessage[Any]`. Conversion is handled at the route layer:

```python
# In OpenCode route handler
message_with_parts: MessageWithParts = ...
chat_message = message_with_parts.to_chat_message()  # or adapter
await session_pool.append_message(session_id, chat_message)
```

The design uses `ChatMessage[Any]` in the API because `StorageProvider` is the canonical persistence layer and it uses `ChatMessage`. OpenCode-specific types are converted at the boundary.

### Helper Methods

```python
    async def get_message_count(self, session_id: str) -> int:
        """Get the number of messages in a session.

        Used by SessionInfo DTO (A7.2) for efficient counting without
        loading all messages.
        """

    async def get_message(
        self,
        session_id: str,
        message_id: str,
    ) -> ChatMessage[Any] | None:
        """Get a single message by ID.

        Returns None if not found.
        """

    # ── Core API Methods ───────────────────────────────────────────

    async def get_messages(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ChatMessage[Any]]:
        """Get message history for a session.

        Args:
            session_id: The session to retrieve messages for.
            limit: Maximum number of messages to return. None means no limit.
            offset: Number of messages to skip (for pagination).

        Returns:
            List of messages ordered by timestamp (oldest first).
        """

    async def append_message(
        self,
        session_id: str,
        message: ChatMessage[Any],
    ) -> str:
        """Append a message to a session's history.

        Args:
            session_id: The session to append to.
            message: The message to append.

        Returns:
            The ID of the appended message.
        """

    async def copy_messages(
        self,
        source_session_id: str,
        target_session_id: str,
        *,
        up_to_message_id: str | None = None,
    ) -> str | None:
        """Copy messages from one session to another.

        Used by share_session (copy all) and revert_session (copy up to
        a specific message).

        Args:
            source_session_id: Session to copy from.
            target_session_id: Session to copy to.
            up_to_message_id: If set, only copy messages up to and
                including this message ID. If None, copy all messages.

        Returns:
            The ID of the fork point message (last copied message),
            or None if no messages were copied.
        """

    async def truncate_messages(
        self,
        session_id: str,
        up_to_message_id: str,
    ) -> int:
        """Truncate messages after a specific message ID.

        Used by revert_session to remove messages after the revert point.

        Args:
            session_id: The session to truncate.
            up_to_message_id: Keep messages up to and including this ID,
                remove everything after.

        Returns:
            Number of messages removed.
        """
```

## Implementation Strategy

### Layered Architecture

```
┌─────────────────────────────────────────┐
│         OpenCode Server Routes          │
│   (share_session, revert_session, etc.) │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│           SessionPool API               │
│   get_messages / append_message / ...   │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│           StorageManager Layer          │
│   AgentPool.storage → SQLStorageProvider│
│   (get_messages, log_message, etc.)     │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         StorageProvider Layer           │
│   SQLStorageProvider / MemoryStorage    │
│   (sqlalchemy queries, in-memory store) │
└─────────────────────────────────────────┘
```

### StorageManager Integration

The existing `StorageProvider` has methods that map to the new API:

- `get_session_messages(session_id, include_ancestors=False)` — maps to `get_messages()` (without pagination)
- `log_message(message=...)` — maps to `append_message()`
- `fork_conversation(source_session_id, new_session_id, ...)` — maps to `copy_messages()`

**Enhancements needed**:
1. Add `truncate_messages()` to `StorageProvider` base class
2. Add pagination support to `StorageProvider.get_session_messages()` (new `limit` parameter)
3. Add pagination forwarding to `StorageManager.get_session_messages()`

```python
# In wolfharness_storage/base.py (StorageProvider)
async def truncate_messages(
    self,
    session_id: str,
    up_to_message_id: str,
) -> int:
    """Remove all messages after the given message ID.

    Keeps messages up to and including up_to_message_id,
    removes everything after it. Used by revert_session.

    Returns the count of removed messages.
    """
    msg = f"{self.__class__.__name__} does not support truncating messages"
    raise NotImplementedError(msg)

async def get_session_messages(
    self,
    session_id: str,
    *,
    include_ancestors: bool = False,
    limit: int | None = None,
) -> list[ChatMessage[Any]]:
    """Get messages with optional pagination.

    Args:
        session_id: The session to retrieve messages for.
        include_ancestors: Whether to include messages from ancestor sessions.
        limit: Maximum number of messages to return (new parameter for Migration B).
    """

# In wolfharness/storage/manager.py (StorageManager)
async def get_session_messages(
    self,
    session_id: str,
    *,
    include_ancestors: bool = False,
    limit: int | None = None,
) -> list[ChatMessage[Any]]:
    """Forward to active StorageProvider with pagination."""
    provider = self.get_history_provider()
    return await provider.get_session_messages(
        session_id,
        include_ancestors=include_ancestors,
        limit=limit,
    )
```

### In-Memory Fallback

During Migration B, before all StorageProviders implement pagination:

```python
class SessionPool:
    async def get_messages(self, session_id: str, ..., limit: int | None = None) -> list[ChatMessage[Any]]:
        # Access storage through AgentPool (accessible via SessionController)
        agent_pool = self.sessions.pool  # SessionController holds AgentPool ref
        storage = agent_pool.storage if agent_pool is not None else None
        if storage is not None:
            try:
                messages = await storage.get_session_messages(session_id, limit=limit)
                return messages
            except (NotImplementedError, TypeError):
                # Fallback: get all and slice
                messages = await storage.get_session_messages(session_id)
                if limit is not None:
                    messages = messages[-limit:]
                return messages
        # No storage provider - this shouldn't happen in production
        return []
```

## Error Handling

| Error | Condition | Handling |
|-------|-----------|----------|
| `KeyError` | `session_id` not found | Raised to caller; caller should create session first |
| `ValueError` | `up_to_message_id` not found in truncate | Raised to caller; indicates invalid revert target |
| `NotImplementedError` | StorageProvider doesn't support operation | Fallback to in-memory cache or raise |

## Sequence Diagram: copy_messages (Session Share)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Client    │     │ OpenCode    │     │   SessionPool   │     │ StorageManager  │     │ StorageProvider │
│             │     │   Server    │     │                 │     │                 │     │   (SQL/Zed)     │
└──────┬──────┘     └──────┬──────┘     └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
       │                   │                     │                       │                       │
       │ POST /share       │                     │                       │                       │
       │──────────────────▶│                     │                       │                       │
       │                   │                     │                       │                       │
       │                   │ copy_messages(      │                       │                       │
       │                   │   from="s1",        │                       │                       │
       │                   │   to="s2")          │                       │                       │
       │                   │────────────────────▶│                       │                       │
       │                   │                     │                       │                       │
       │                   │                     │ fork_conversation(    │                       │
       │                   │                     │   source="s1",        │                       │
       │                   │                     │   new="s2")           │                       │
       │                   │                     │──────────────────────▶│                       │
       │                   │                     │                       │                       │
       │                   │                     │                       │ fork_conversation(    │
       │                   │                     │                       │   source="s1",        │
       │                   │                     │                       │   new="s2")           │
       │                   │                     │                       │──────────────────────▶│
       │                   │                     │                       │                       │
       │                   │                     │                       │◀──────────────────────│
       │                   │                     │                       │ fork_point_msg_id     │
       │                   │                     │                       │                       │
       │                   │                     │◀──────────────────────│                       │
       │                   │                     │ fork_point_msg_id     │                       │
       │                   │                     │                       │                       │
       │                   │◀────────────────────│ return fork_point     │                       │
       │                   │                     │                       │                       │
       │ 200 OK            │                     │                       │                       │
       │◀──────────────────│                     │                       │                       │
```

## Migration Path

### Phase 1: Add API to SessionPool
- Add the 4 core methods to SessionPool
- Add `truncate_messages()` to StorageProvider base class
- Implement in-memory fallback
- Add unit tests

### Phase 2: Update OpenCode Routes
- `share_session()` → use `SessionPool.copy_messages()`
- `revert_session()` → use `SessionPool.truncate_messages()`
- `get_or_load_session()` → use `SessionPool.get_messages()`
- Session fork → use `SessionPool.copy_messages()`

### Phase 3: Remove ServerState.messages
- After all routes are migrated
- Delete `ServerState.messages` dictionary
- Delete `ServerState.append_message()` method
- Update tests

## Open Questions

1. **Message ordering in concurrent scenarios**: If two turns append messages simultaneously, does SessionPool need ordering guarantees beyond StorageProvider's transaction isolation?
   - *Answer*: SessionPool.turn_lock already serializes turns per session, so concurrent appends to the same session are impossible.

2. **Message ID format**: Should message IDs be UUIDs, ULIDs, or monotonic counters?
   - *Recommendation*: ULIDs for time-sortable, lexicographically ordered IDs that support efficient range queries.

3. **Large message histories**: What if a session has 10,000+ messages?
   - *Answer*: `get_messages()` supports pagination via `limit` / `offset`. StorageProvider implementations should use indexed queries.
