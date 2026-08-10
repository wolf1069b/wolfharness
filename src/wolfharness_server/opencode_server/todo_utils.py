"""Helpers for presenting todo tracker state to OpenCode clients."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from wolfharness.utils.todos import TodoPriority, TodoStatus, TodoTracker


class TodoModelFactory[TodoModelT](Protocol):
    """Factory protocol shared by OpenCode todo model classes."""

    def __call__(
        self,
        *,
        id: str,  # noqa: A002 - mirrors OpenCode's todo model field name.
        content: str,
        status: TodoStatus,
        priority: TodoPriority,
    ) -> TodoModelT: ...


def build_opencode_todos[TodoModelT](
    tracker: TodoTracker,
    todo_model: TodoModelFactory[TodoModelT],
) -> list[TodoModelT]:
    """Build OpenCode todo models from tracked todo entries."""
    return [
        todo_model(id=e.id, content=e.content, status=e.status, priority=e.priority)
        for e in tracker.entries
    ]
