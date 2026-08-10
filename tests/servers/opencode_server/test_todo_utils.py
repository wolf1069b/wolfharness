"""Tests for OpenCode todo presentation helpers."""

from __future__ import annotations

import pytest

from wolfharness.utils.todos import TodoTracker
from wolfharness_server.opencode_server.models.session import Todo
from wolfharness_server.opencode_server.todo_utils import build_opencode_todos


pytestmark = pytest.mark.integration


def test_build_opencode_todos_uses_real_tracker_entries_only() -> None:
    """Test only real todo entries are exposed to OpenCode."""
    tracker = TodoTracker()
    tracker.add("First task")
    tracker.add("Second task")

    todos = build_opencode_todos(tracker, Todo)

    assert len(todos) == 2
    assert todos[0].content == "First task"
    assert todos[1].content == "Second task"
    assert {todo.id for todo in todos} == {"todo_1", "todo_2"}
