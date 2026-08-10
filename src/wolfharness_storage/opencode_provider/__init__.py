"""OpenCode SQLite storage provider.

This package implements the storage backend compatible with OpenCode's
SQLite database format (>= 1.2).

The database is typically at ~/.local/share/opencode/opencode.db and contains:
- project: Project/worktree information
- session: Conversation sessions
- message: Messages within sessions (data stored as JSON)
- part: Message parts (data stored as JSON)
- todo: Session todos
"""

from __future__ import annotations

from wolfharness_storage.opencode_provider.provider import OpenCodeStorageProvider

__all__ = [
    "OpenCodeStorageProvider",
]
