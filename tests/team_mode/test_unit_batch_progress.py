"""Unit tests for FileTeamState batch creation and progress tracking.

Covers: create_tasks_batch with #N and symbolic references, atomic
failure scenarios, progress tracking via update_task, TaskRecord,
format_task_xml, and format_owner_summary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wolfharness.capabilities.file_team_state import (
    FileTeamState,
    TaskRecord,
    format_owner_summary,
    format_task_xml,
)


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def state(tmp_path: Path) -> FileTeamState:
    """Return a FileTeamState rooted in a temp directory."""
    return FileTeamState(str(tmp_path))


@pytest.fixture
def initialized_team(state: FileTeamState) -> FileTeamState:
    """Return a FileTeamState with a team already initialized."""
    state.init(
        team_id="team-1",
        team_name="Test Team",
        members=[{"name": "alice", "agent": "alice"}, {"name": "bob"}],
    )
    return state


# ---------------------------------------------------------------------------
# 8.6 Batch creation tests
# ---------------------------------------------------------------------------


def test_batch_with_hash_dependencies(initialized_team: FileTeamState) -> None:
    """Given: initialized team.

    When: create_tasks_batch with #N dependencies.
    Then: tasks created with resolved dependency IDs.
    """
    task_ids = initialized_team.create_tasks_batch(
        "team-1",
        [
            {"subject": "Task A"},
            {"subject": "Task B", "blocked_by": ["#0"]},
        ],
    )

    assert len(task_ids) == 2
    tasks = initialized_team.list_tasks("team-1")
    task_b = next(t for t in tasks if t["subject"] == "Task B")
    assert task_b["blocked_by"] == [task_ids[0]]


def test_batch_with_symbolic_id_dependencies(initialized_team: FileTeamState) -> None:
    """Given: initialized team.

    When: create_tasks_batch with symbolic id dependencies.
    Then: tasks created with resolved symbolic references.
    """
    task_ids = initialized_team.create_tasks_batch(
        "team-1",
        [
            {"subject": "Research", "id": "research"},
            {"subject": "Analyze", "id": "analyze", "blocked_by": ["research"]},
        ],
    )

    assert len(task_ids) == 2
    tasks = initialized_team.list_tasks("team-1")
    analyze_task = next(t for t in tasks if t["subject"] == "Analyze")
    assert analyze_task["blocked_by"] == [task_ids[0]]


def test_batch_with_symbolic_parent_id(initialized_team: FileTeamState) -> None:
    """Given: initialized team.

    When: create_tasks_batch with symbolic parent_id.
    Then: child task created with resolved parent reference.
    """
    task_ids = initialized_team.create_tasks_batch(
        "team-1",
        [
            {"subject": "Parent", "id": "parent"},
            {"subject": "Child", "parent_id": "parent"},
        ],
    )

    assert len(task_ids) == 2
    child = initialized_team.get_task("team-1", task_ids[1])
    assert child is not None
    assert child["parent_id"] == task_ids[0]


def test_batch_unresolved_symbolic_reference_passes_through(
    initialized_team: FileTeamState,
) -> None:
    """Given: initialized team.

    When: create_tasks_batch with unresolved symbolic ref in blocked_by.
    Then: ref passes through as-is (assumed to be existing task ID).
    """
    task_ids = initialized_team.create_tasks_batch(
        "team-1",
        [{"subject": "Task", "blocked_by": ["nonexistent_sym"]}],
    )

    assert len(task_ids) == 1
    task = initialized_team.get_task("team-1", task_ids[0])
    assert task is not None
    assert task["blocked_by"] == ["nonexistent_sym"]


def test_batch_invalid_hash_index_atomic_failure(
    initialized_team: FileTeamState,
) -> None:
    """Given: initialized team.

    When: create_tasks_batch with #99 out-of-range reference.
    Then: ValueError raised, no tasks created.
    """
    with pytest.raises(ValueError, match="out of range"):
        initialized_team.create_tasks_batch(
            "team-1",
            [
                {"subject": "Task A"},
                {"subject": "Task B", "blocked_by": ["#99"]},
            ],
        )

    # Verify no tasks were created (atomic failure).
    tasks = initialized_team.list_tasks("team-1")
    assert len(tasks) == 0


def test_batch_missing_subject_atomic_failure(
    initialized_team: FileTeamState,
) -> None:
    """Given: initialized team.

    When: create_tasks_batch with a task missing 'subject'.
    Then: ValueError raised, no tasks created.
    """
    with pytest.raises(ValueError, match="missing required 'subject'"):
        initialized_team.create_tasks_batch(
            "team-1",
            [{"description": "No subject"}],
        )

    tasks = initialized_team.list_tasks("team-1")
    assert len(tasks) == 0


def test_batch_exceeding_max_tasks_atomic_failure(
    initialized_team: FileTeamState,
) -> None:
    """Given: initialized team with tasks near the 100 limit.

    When: create_tasks_batch would exceed _MAX_TASKS.
    Then: ValueError raised with max tasks message.
    """
    from wolfharness.capabilities.file_team_state import _MAX_TASKS

    for i in range(_MAX_TASKS - 1):
        initialized_team.create_task("team-1", {"subject": f"Filler {i}"})

    with pytest.raises(ValueError, match="max tasks limit"):
        initialized_team.create_tasks_batch(
            "team-1",
            [{"subject": "A"}, {"subject": "B"}],
        )


def test_batch_with_progress_total(initialized_team: FileTeamState) -> None:
    """Given: initialized team.

    When: create_tasks_batch with progress_total on a task.
    Then: task created with progress_total stored.
    """
    task_ids = initialized_team.create_tasks_batch(
        "team-1",
        [{"subject": "Tracked task", "progress_total": 50}],
    )

    assert len(task_ids) == 1
    task = initialized_team.get_task("team-1", task_ids[0])
    assert task is not None
    assert task["progress_total"] == 50


def test_batch_empty_list_returns_empty(initialized_team: FileTeamState) -> None:
    """Given: initialized team.

    When: create_tasks_batch with empty list.
    Then: returns empty list (no error).
    """
    result = initialized_team.create_tasks_batch("team-1", [])
    assert result == []


def test_batch_self_reference_atomic_failure(
    initialized_team: FileTeamState,
) -> None:
    """Given: initialized team.

    When: create_tasks_batch with a task referencing itself via #N.
    Then: ValueError raised with "cannot reference itself".
    """
    with pytest.raises(ValueError, match="cannot reference itself"):
        initialized_team.create_tasks_batch(
            "team-1",
            [{"subject": "Self-ref", "blocked_by": ["#0"]}],
        )


def test_batch_duplicate_symbolic_id_atomic_failure(
    initialized_team: FileTeamState,
) -> None:
    """Given: initialized team.

    When: create_tasks_batch with duplicate symbolic id.
    Then: ValueError raised with "Duplicate symbolic id".
    """
    with pytest.raises(ValueError, match="Duplicate symbolic id"):
        initialized_team.create_tasks_batch(
            "team-1",
            [
                {"subject": "A", "id": "dup"},
                {"subject": "B", "id": "dup"},
            ],
        )


# ---------------------------------------------------------------------------
# 9.7 Progress tracking tests (via FileTeamState.update_task)
# ---------------------------------------------------------------------------


def test_update_task_set_progress(initialized_team: FileTeamState) -> None:
    """Given: existing task.

    When: update_task called with progress_current=3, progress_total=10.
    Then: task stored with both progress values.
    """
    task_id = initialized_team.create_task("team-1", {"subject": "Task"})
    result = initialized_team.update_task(
        "team-1",
        task_id,
        {},
        progress_current=3,
        progress_total=10,
    )

    assert result["progress_current"] == 3
    assert result["progress_total"] == 10


def test_update_task_progress_current_only_preserves_total(
    initialized_team: FileTeamState,
) -> None:
    """Given: task with progress_total=10.

    When: update_task called with progress_current=5 only.
    Then: total preserved, current updated.
    """
    task_id = initialized_team.create_task("team-1", {"subject": "Task"})
    initialized_team.update_task("team-1", task_id, {}, progress_current=2, progress_total=10)

    result = initialized_team.update_task("team-1", task_id, {}, progress_current=5)

    assert result["progress_current"] == 5
    assert result["progress_total"] == 10


# ---------------------------------------------------------------------------
# TaskRecord, format_task_xml, format_owner_summary
# ---------------------------------------------------------------------------


def test_task_record_from_dict_with_progress() -> None:
    """Given: a raw task dict with progress fields.

    When: TaskRecord.from_dict is called.
    Then: TaskRecord has correct progress values.
    """
    record = TaskRecord.from_dict({
        "task_id": "t1",
        "subject": "Test",
        "progress_current": 5,
        "progress_total": 10,
    })

    assert record.task_id == "t1"
    assert record.subject == "Test"
    assert record.progress_current == 5
    assert record.progress_total == 10


def test_task_record_from_dict_defaults() -> None:
    """Given: a minimal task dict.

    When: TaskRecord.from_dict is called.
    Then: defaults are applied correctly.
    """
    record = TaskRecord.from_dict({"task_id": "t1", "subject": "Test"})

    assert record.description == ""
    assert record.owner == ""
    assert record.status == "pending"
    assert record.blocked_by == []
    assert record.parent_id is None
    assert record.children == []
    assert record.is_unblocked is True
    assert record.last_note == ""
    assert record.progress_current is None
    assert record.progress_total is None


def test_format_task_xml_with_progress() -> None:
    """Given: TaskRecord with progress_current=3, progress_total=10.

    When: format_task_xml is called.
    Then: XML includes progress="3/10" attribute.
    """
    record = TaskRecord(
        task_id="t1",
        subject="Test task",
        progress_current=3,
        progress_total=10,
    )

    xml = format_task_xml(record)

    assert 'progress="3/10"' in xml
    assert 'id="t1"' in xml
    assert 'status="pending"' in xml
    assert 'owner=""' in xml
    assert "Test task" in xml


def test_format_task_xml_without_progress() -> None:
    """Given: TaskRecord without progress values.

    When: format_task_xml is called.
    Then: XML does NOT include progress attribute.
    """
    record = TaskRecord(task_id="t1", subject="No progress task")

    xml = format_task_xml(record)

    assert "progress=" not in xml
    assert "No progress task" in xml


def test_format_task_xml_with_note() -> None:
    """Given: TaskRecord with last_note set and progress values.

    When: format_task_xml is called.
    Then: XML includes 'note: <content>' line.
    """
    record = TaskRecord(
        task_id="t1",
        subject="Task",
        last_note="Important note",
        progress_current=1,
        progress_total=5,
    )

    xml = format_task_xml(record)

    assert "note: Important note" in xml


def test_format_task_xml_blocked() -> None:
    """Given: TaskRecord with is_unblocked=False and progress values.

    When: format_task_xml is called.
    Then: XML includes blocked="true" attribute.
    """
    record = TaskRecord(
        task_id="t1",
        subject="Blocked task",
        is_unblocked=False,
        progress_current=0,
        progress_total=10,
    )

    xml = format_task_xml(record)

    assert 'blocked="true"' in xml


def test_format_owner_summary_with_tasks() -> None:
    """Given: list of TaskRecords with different owners.

    When: format_owner_summary is called.
    Then: returns correct summary string.
    """
    tasks = [
        TaskRecord(task_id="t1", subject="A", owner="researcher"),
        TaskRecord(task_id="t2", subject="B", owner="researcher"),
        TaskRecord(task_id="t3", subject="C", owner="analyst"),
        TaskRecord(task_id="t4", subject="D", owner=""),
    ]

    summary = format_owner_summary(tasks)

    assert "4 tasks" in summary
    assert "researcher=2" in summary
    assert "analyst=1" in summary
    assert "unassigned=1" in summary


def test_format_owner_summary_empty() -> None:
    """Given: empty list of tasks.

    When: format_owner_summary is called.
    Then: returns "0 tasks".
    """
    summary = format_owner_summary([])

    assert summary == "0 tasks"
