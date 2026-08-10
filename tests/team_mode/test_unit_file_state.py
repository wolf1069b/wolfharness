"""Unit tests for FileTeamState.

All tests use ``tmp_path`` for isolation and are marked ``@pytest.mark.unit``.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
from typing import TYPE_CHECKING

import pytest

from wolfharness.capabilities.file_team_state import (
    FileTeamState,
    TaskRecord,
    format_task_xml,
    start_team_cleanup_task,
)


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.unit


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# 1. init creates correct directory structure
# ------------------------------------------------------------------


def test_init_creates_directory_structure(state: FileTeamState, tmp_path: Path) -> None:
    """Given init, the full directory tree and state.json exist."""
    state.init("team-1", "Test Team", [{"name": "alice"}])

    team_dir = tmp_path / "teams" / "team-1"
    assert team_dir.is_dir()
    assert (team_dir / "state.json").is_file()
    assert (team_dir / "inboxes").is_dir()
    assert (team_dir / "tasks").is_dir()
    assert (team_dir / "blackboard").is_dir()
    assert (team_dir / "blackboard" / ".locks").is_dir()


def test_init_state_json_contents(state: FileTeamState) -> None:
    """Given init, state.json has correct metadata fields."""
    state.init("team-1", "My Team", [{"name": "alice"}, {"name": "bob"}])

    state_data = json.loads(
        (state._team_dir("team-1") / "state.json").read_text(),
    )
    assert state_data["team_name"] == "My Team"
    assert state_data["status"] == "active"
    assert state_data["ended_at"] is None
    assert "created_at" in state_data
    assert "alice" in state_data["members"]
    assert "bob" in state_data["members"]


# ------------------------------------------------------------------
# 2. register_member + get_member_session_id round-trip
# ------------------------------------------------------------------


def test_register_and_get_session_id(initialized_team: FileTeamState) -> None:
    """Given register_member, get_member_session_id returns the value."""
    initialized_team.register_member("team-1", "alice", "sess-123")

    result = initialized_team.get_member_session_id("team-1", "alice")
    assert result == "sess-123"


def test_get_member_session_id_returns_none_for_unregistered(
    initialized_team: FileTeamState,
) -> None:
    """Given an unregistered member, get_member_session_id returns None."""
    result = initialized_team.get_member_session_id("team-1", "charlie")
    assert result is None


# ------------------------------------------------------------------
# 3. write_message + read_messages round-trip
# ------------------------------------------------------------------


def test_write_and_read_messages(initialized_team: FileTeamState) -> None:
    """Given write_message, read_messages returns the messages sorted."""
    msg1 = {"from": "bob", "content": "hello", "timestamp": "2026-01-01T00:00:00Z"}
    msg2 = {"from": "bob", "content": "world", "timestamp": "2026-01-02T00:00:00Z"}

    initialized_team.write_message("team-1", "alice", msg2)
    initialized_team.write_message("team-1", "alice", msg1)

    messages = initialized_team.read_messages("team-1", "alice")
    assert len(messages) == 2
    assert messages[0]["content"] == "hello"
    assert messages[1]["content"] == "world"


# ------------------------------------------------------------------
# 4. read_messages returns empty list for non-existent member
# ------------------------------------------------------------------


def test_read_messages_empty_for_nonexistent_member(
    initialized_team: FileTeamState,
) -> None:
    """Given a member with no inbox, read_messages returns []."""
    result = initialized_team.read_messages("team-1", "nonexistent")
    assert result == []


# ------------------------------------------------------------------
# 5. create_task returns task_id
# ------------------------------------------------------------------


def test_create_task_returns_task_id(initialized_team: FileTeamState) -> None:
    """Given create_task, a UUID task_id is returned."""
    task_id = initialized_team.create_task(
        "team-1",
        {"title": "Do something"},
    )
    assert isinstance(task_id, str)
    assert len(task_id) > 0


# ------------------------------------------------------------------
# 6. list_tasks returns all tasks
# ------------------------------------------------------------------


def test_list_tasks_returns_all_tasks(initialized_team: FileTeamState) -> None:
    """Given multiple tasks, list_tasks returns all of them."""
    initialized_team.create_task("team-1", {"title": "task-a"})
    initialized_team.create_task("team-1", {"title": "task-b"})

    tasks = initialized_team.list_tasks("team-1")
    assert len(tasks) == 2
    titles = {t["title"] for t in tasks}
    assert titles == {"task-a", "task-b"}


# ------------------------------------------------------------------
# 7. update_task merges updates
# ------------------------------------------------------------------


def test_update_task_merges_updates(initialized_team: FileTeamState) -> None:
    """Given update_task, fields are merged into the existing task."""
    task_id = initialized_team.create_task(
        "team-1",
        {"title": "original", "status": "pending"},
    )
    updated = initialized_team.update_task(
        "team-1",
        task_id,
        {"status": "completed"},
    )
    assert updated["title"] == "original"
    assert updated["status"] == "completed"
    assert updated["task_id"] == task_id


# ------------------------------------------------------------------
# 8. read_blackboard returns value with version=1 on first write
# ------------------------------------------------------------------


def test_read_blackboard_returns_version_1(initialized_team: FileTeamState) -> None:
    """Given a first write, read_blackboard returns version=1."""
    initialized_team.write_blackboard(
        "team-1",
        "context",
        {"data": "hello"},
        written_by="alice",
    )
    result = initialized_team.read_blackboard("team-1", "context")
    assert result is not None
    assert result["version"] == 1
    assert result["value"] == {"data": "hello"}
    assert result["written_by"] == "alice"
    assert "written_at" in result


# ------------------------------------------------------------------
# 9. write_blackboard with correct expected_version succeeds
# ------------------------------------------------------------------


def test_write_blackboard_correct_version_succeeds(
    initialized_team: FileTeamState,
) -> None:
    """Given matching expected_version, the write succeeds and increments."""
    initialized_team.write_blackboard("team-1", "key1", {"v": 1})
    result = initialized_team.write_blackboard(
        "team-1",
        "key1",
        {"v": 2},
        expected_version=1,
        written_by="bob",
    )
    assert result == "Written, version=2"

    entry = initialized_team.read_blackboard("team-1", "key1")
    assert entry is not None
    assert entry["version"] == 2
    assert entry["value"] == {"v": 2}


# ------------------------------------------------------------------
# 10. write_blackboard with wrong expected_version returns "Conflict"
# ------------------------------------------------------------------


def test_write_blackboard_wrong_version_returns_conflict(
    initialized_team: FileTeamState,
) -> None:
    """Given a mismatched expected_version, write returns conflict."""
    initialized_team.write_blackboard("team-1", "key1", {"v": 1})
    result = initialized_team.write_blackboard(
        "team-1",
        "key1",
        {"v": 2},
        expected_version=99,
    )
    assert result == "Conflict: current version is 1"


# ------------------------------------------------------------------
# 11. list_blackboard returns all keys
# ------------------------------------------------------------------


def test_list_blackboard_returns_all_keys(initialized_team: FileTeamState) -> None:
    """Given multiple keys, list_blackboard returns them sorted."""
    initialized_team.write_blackboard("team-1", "zebra", {"n": 1})
    initialized_team.write_blackboard("team-1", "alpha", {"n": 2})
    initialized_team.write_blackboard("team-1", "mid_key", {"n": 3})

    keys = initialized_team.list_blackboard("team-1")
    assert keys == ["alpha", "mid_key", "zebra"]


def test_list_blackboard_returns_namespace_keys(initialized_team: FileTeamState) -> None:
    """Given keys with '/' separators (namespace keys), list_blackboard returns them all.

    Keys like ``findings/artisan`` are stored as nested files
    (``blackboard/findings/artisan.json``).  ``list_blackboard`` must
    use ``rglob`` to discover them, not just top-level ``glob``.
    """
    initialized_team.write_blackboard("team-1", "principles", {"v": 1})
    initialized_team.write_blackboard("team-1", "findings/artisan", {"v": 2})
    initialized_team.write_blackboard("team-1", "findings/curator", {"v": 3})
    initialized_team.write_blackboard("team-1", "problems/critic", {"v": 4})

    keys = initialized_team.list_blackboard("team-1")
    assert keys == [
        "findings/artisan",
        "findings/curator",
        "principles",
        "problems/critic",
    ]


# ------------------------------------------------------------------
# 12. delete_blackboard removes key
# ------------------------------------------------------------------


def test_delete_blackboard_removes_key(initialized_team: FileTeamState) -> None:
    """Given delete_blackboard, the key is removed."""
    initialized_team.write_blackboard("team-1", "temp", {"x": 1})
    assert initialized_team.read_blackboard("team-1", "temp") is not None

    initialized_team.delete_blackboard("team-1", "temp")
    assert initialized_team.read_blackboard("team-1", "temp") is None


# ------------------------------------------------------------------
# 13. path traversal key rejected
# ------------------------------------------------------------------


def test_path_traversal_key_rejected(initialized_team: FileTeamState) -> None:
    """Given a traversal key, ValueError is raised."""
    # The key contains dots which fail the regex check first — both
    # checks reject path traversal; the regex is the first gate.
    with pytest.raises(ValueError, match=r"(Invalid blackboard key|Path traversal)"):
        initialized_team.read_blackboard("team-1", "../../../etc/passwd")


def test_invalid_key_characters_rejected(initialized_team: FileTeamState) -> None:
    """Given a key with invalid characters, ValueError is raised."""
    with pytest.raises(ValueError, match="Invalid blackboard key"):
        initialized_team.write_blackboard("team-1", "key with spaces", {"x": 1})


# ------------------------------------------------------------------
# 14. task dependency: blocked task becomes unblocked when dep completes
# ------------------------------------------------------------------


def test_task_unblocked_when_dep_completes(initialized_team: FileTeamState) -> None:
    """Given a completed dependency, the blocked task is unblocked."""
    dep_id = initialized_team.create_task(
        "team-1",
        {"title": "dependency", "status": "pending"},
    )
    initialized_team.create_task(
        "team-1",
        {"title": "blocked", "status": "pending", "blocked_by": [dep_id]},
    )

    tasks = initialized_team.list_tasks("team-1")
    blocked_task = next(t for t in tasks if t["title"] == "blocked")
    assert blocked_task["is_unblocked"] is False

    initialized_team.update_task("team-1", dep_id, {"status": "completed"})

    tasks = initialized_team.list_tasks("team-1")
    blocked_task = next(t for t in tasks if t["title"] == "blocked")
    assert blocked_task["is_unblocked"] is True


# ------------------------------------------------------------------
# 15. task dependency: blocked task stays blocked when dep fails
# ------------------------------------------------------------------


def test_task_stays_blocked_when_dep_fails(initialized_team: FileTeamState) -> None:
    """Given a failed dependency, the blocked task stays blocked."""
    dep_id = initialized_team.create_task(
        "team-1",
        {"title": "dependency", "status": "pending"},
    )
    initialized_team.create_task(
        "team-1",
        {"title": "blocked", "status": "pending", "blocked_by": [dep_id]},
    )

    initialized_team.update_task("team-1", dep_id, {"status": "failed"})

    tasks = initialized_team.list_tasks("team-1")
    blocked_task = next(t for t in tasks if t["title"] == "blocked")
    assert blocked_task["is_unblocked"] is False


# ------------------------------------------------------------------
# 16. cleanup removes team directory
# ------------------------------------------------------------------


def test_cleanup_removes_team_directory(
    initialized_team: FileTeamState,
    tmp_path: Path,
) -> None:
    """Given cleanup, the team directory is removed."""
    team_dir = tmp_path / "teams" / "team-1"
    assert team_dir.exists()

    initialized_team.cleanup("team-1")
    assert not team_dir.exists()


# ------------------------------------------------------------------
# 17. cleanup_expired_teams removes old deleted teams
# ------------------------------------------------------------------


def test_cleanup_expired_teams_removes_old_deleted(
    state: FileTeamState,
    tmp_path: Path,
) -> None:
    """Given a deleted team older than TTL, it is removed."""
    state.init("old-team", "Old", [{"name": "alice"}])

    # Mark as deleted with an old ended_at timestamp.
    state_path = tmp_path / "teams" / "old-team" / "state.json"
    state_data = json.loads(state_path.read_text())
    state_data["status"] = "deleted"
    old_time = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=100)).isoformat()
    state_data["ended_at"] = old_time
    state_path.write_text(json.dumps(state_data, default=str))

    removed = FileTeamState.cleanup_expired_teams(str(tmp_path), ttl_hours=24)
    assert removed == 1
    assert not (tmp_path / "teams" / "old-team").exists()


# ------------------------------------------------------------------
# 18. cleanup_expired_teams preserves active teams
# ------------------------------------------------------------------


def test_cleanup_expired_teams_preserves_active(
    state: FileTeamState,
    tmp_path: Path,
) -> None:
    """Given an active team, it is not removed by cleanup_expired_teams."""
    state.init("active-team", "Active", [{"name": "alice"}])

    removed = FileTeamState.cleanup_expired_teams(str(tmp_path), ttl_hours=1)
    assert removed == 0
    assert (tmp_path / "teams" / "active-team").exists()


# ------------------------------------------------------------------
# 19. task with no blocked_by is always unblocked
# ------------------------------------------------------------------


def test_task_without_blocked_by_is_unblocked(
    initialized_team: FileTeamState,
) -> None:
    """Given a task with no blocked_by, is_unblocked is True."""
    initialized_team.create_task("team-1", {"title": "free"})

    tasks = initialized_team.list_tasks("team-1")
    assert tasks[0]["is_unblocked"] is True


# ------------------------------------------------------------------
# 20. write_blackboard without expected_version always succeeds
# ------------------------------------------------------------------


def test_write_blackboard_no_expected_version_always_succeeds(
    initialized_team: FileTeamState,
) -> None:
    """Given no expected_version, writes always succeed (no version check)."""
    initialized_team.write_blackboard("team-1", "key", {"v": 1})
    result = initialized_team.write_blackboard("team-1", "key", {"v": 2})
    assert result == "Written, version=2"

    result = initialized_team.write_blackboard("team-1", "key", {"v": 3})
    assert result == "Written, version=3"


# ------------------------------------------------------------------
# 21. cleanup marks orphaned active teams
# ------------------------------------------------------------------


def test_cleanup_marks_orphaned_active_old(
    state: FileTeamState,
    tmp_path: Path,
) -> None:
    """Given an active team older than TTL with no ended_at, it is marked orphaned."""
    state.init("stale-team", "Stale", [{"name": "alice"}])

    state_path = tmp_path / "teams" / "stale-team" / "state.json"
    state_data = json.loads(state_path.read_text())
    old_time = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=100)).isoformat()
    state_data["created_at"] = old_time
    state_data["ended_at"] = None
    state_path.write_text(json.dumps(state_data, default=str))

    removed = FileTeamState.cleanup_expired_teams(str(tmp_path), ttl_hours=24)
    assert removed == 0
    assert (tmp_path / "teams" / "stale-team").exists()

    updated = json.loads(state_path.read_text())
    assert updated["status"] == "orphaned"


# ------------------------------------------------------------------
# 22. cleanup preserves active recent teams
# ------------------------------------------------------------------


def test_cleanup_preserves_active_recent(
    state: FileTeamState,
    tmp_path: Path,
) -> None:
    """Given a recently created active team, it is not modified by cleanup."""
    state.init("fresh-team", "Fresh", [{"name": "alice"}])

    removed = FileTeamState.cleanup_expired_teams(str(tmp_path), ttl_hours=24)
    assert removed == 0
    assert (tmp_path / "teams" / "fresh-team").exists()

    state_path = tmp_path / "teams" / "fresh-team" / "state.json"
    state_data = json.loads(state_path.read_text())
    assert state_data["status"] == "active"


# ------------------------------------------------------------------
# 23. cleanup returns correct count of removed directories
# ------------------------------------------------------------------


def test_cleanup_returns_correct_count(
    state: FileTeamState,
    tmp_path: Path,
) -> None:
    """Given multiple teams (some expired, some not), cleanup returns correct count."""
    # Team 1: deleted and old -> removed
    state.init("old-deleted", "Old", [{"name": "alice"}])
    sp = tmp_path / "teams" / "old-deleted" / "state.json"
    sd = json.loads(sp.read_text())
    sd["status"] = "deleted"
    sd["ended_at"] = (
        datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=100)
    ).isoformat()
    sp.write_text(json.dumps(sd, default=str))

    # Team 2: deleted but recent -> preserved
    state.init("recent-deleted", "Recent", [{"name": "alice"}])
    sp = tmp_path / "teams" / "recent-deleted" / "state.json"
    sd = json.loads(sp.read_text())
    sd["status"] = "deleted"
    sd["ended_at"] = datetime.datetime.now(datetime.UTC).isoformat()
    sp.write_text(json.dumps(sd, default=str))

    # Team 3: active and old -> orphaned, preserved
    state.init("old-active", "OldActive", [{"name": "alice"}])
    sp = tmp_path / "teams" / "old-active" / "state.json"
    sd = json.loads(sp.read_text())
    sd["created_at"] = (
        datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=100)
    ).isoformat()
    sp.write_text(json.dumps(sd, default=str))

    removed = FileTeamState.cleanup_expired_teams(str(tmp_path), ttl_hours=24)
    assert removed == 1
    assert not (tmp_path / "teams" / "old-deleted").exists()
    assert (tmp_path / "teams" / "recent-deleted").exists()
    assert (tmp_path / "teams" / "old-active").exists()


# ------------------------------------------------------------------
# 24. cleanup does not orphan active team with ended_at set
# ------------------------------------------------------------------


def test_cleanup_does_not_orphan_with_ended_at(
    state: FileTeamState,
    tmp_path: Path,
) -> None:
    """Given an old active team with ended_at set, it is not orphaned."""
    state.init("ended-team", "Ended", [{"name": "alice"}])

    state_path = tmp_path / "teams" / "ended-team" / "state.json"
    state_data = json.loads(state_path.read_text())
    old_time = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=100)).isoformat()
    state_data["created_at"] = old_time
    state_data["ended_at"] = old_time
    state_data["status"] = "active"
    state_path.write_text(json.dumps(state_data, default=str))

    removed = FileTeamState.cleanup_expired_teams(str(tmp_path), ttl_hours=24)
    assert removed == 0

    updated = json.loads(state_path.read_text())
    assert updated["status"] == "active"


# ------------------------------------------------------------------
# 25. start_team_cleanup_task returns a cancellable task
# ------------------------------------------------------------------


async def test_start_team_cleanup_task_cancellable(
    state: FileTeamState,
    tmp_path: Path,
) -> None:
    """Given start_team_cleanup_task, the returned task can be cancelled."""
    state.init("active-team-2", "Active", [{"name": "alice"}])

    task = await start_team_cleanup_task(
        base_dir=str(tmp_path),
        ttl_hours=24,
        interval_minutes=1,
    )
    assert not task.done()

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert task.done()
    assert task.cancelled()


# ------------------------------------------------------------------
# 26. cleanup_expired_teams returns 0 for non-existent directory
# ------------------------------------------------------------------


def test_cleanup_returns_zero_for_nonexistent_dir(tmp_path: Path) -> None:
    """Given a non-existent teams directory, cleanup returns 0."""
    removed = FileTeamState.cleanup_expired_teams(
        str(tmp_path / "nonexistent"),
        ttl_hours=24,
    )
    assert removed == 0


# ------------------------------------------------------------------
# 27. create_task with parent_id stores it correctly
# ------------------------------------------------------------------


def test_create_task_with_parent_id(initialized_team: FileTeamState) -> None:
    """Given a parent task, creating a child task with parent_id stores it."""
    parent_id = initialized_team.create_task(
        "team-1",
        {"subject": "parent task"},
    )
    child_id = initialized_team.create_task(
        "team-1",
        {"subject": "child task", "parent_id": parent_id},
    )
    child = initialized_team.get_task("team-1", child_id)
    assert child is not None
    assert child["parent_id"] == parent_id
    assert child["task_id"] == child_id


# ------------------------------------------------------------------
# 28. create_task with invalid parent_id raises ValueError
# ------------------------------------------------------------------


def test_create_task_invalid_parent_id_raises(initialized_team: FileTeamState) -> None:
    """Given a non-existent parent_id, create_task raises ValueError."""
    with pytest.raises(ValueError, match="Parent task not found"):
        initialized_team.create_task(
            "team-1",
            {"subject": "orphan", "parent_id": "task_nonexistent"},
        )


# ------------------------------------------------------------------
# 29. get_task returns task or None
# ------------------------------------------------------------------


def test_get_task_returns_task(initialized_team: FileTeamState) -> None:
    """Given an existing task, get_task returns it."""
    task_id = initialized_team.create_task("team-1", {"subject": "find me"})
    result = initialized_team.get_task("team-1", task_id)
    assert result is not None
    assert result["task_id"] == task_id
    assert result["subject"] == "find me"


def test_get_task_returns_none_for_missing(initialized_team: FileTeamState) -> None:
    """Given a non-existent task_id, get_task returns None."""
    result = initialized_team.get_task("team-1", "task_nope")
    assert result is None


# ------------------------------------------------------------------
# 30. list_tasks computes children field
# ------------------------------------------------------------------


def test_list_tasks_computes_children(initialized_team: FileTeamState) -> None:
    """Given parent and child tasks, list_tasks computes children field."""
    parent_id = initialized_team.create_task("team-1", {"subject": "parent"})
    child1_id = initialized_team.create_task(
        "team-1", {"subject": "child1", "parent_id": parent_id}
    )
    child2_id = initialized_team.create_task(
        "team-1", {"subject": "child2", "parent_id": parent_id}
    )

    tasks = initialized_team.list_tasks("team-1")
    parent = next(t for t in tasks if t["task_id"] == parent_id)
    assert set(parent["children"]) == {child1_id, child2_id}

    child1 = next(t for t in tasks if t["task_id"] == child1_id)
    assert child1["children"] == []


# ------------------------------------------------------------------
# 31. list_children returns only direct children
# ------------------------------------------------------------------


def test_list_children_returns_direct_children(
    initialized_team: FileTeamState,
) -> None:
    """Given nested tasks, list_children returns only direct children."""
    parent_id = initialized_team.create_task("team-1", {"subject": "parent"})
    child_id = initialized_team.create_task("team-1", {"subject": "child", "parent_id": parent_id})
    initialized_team.create_task("team-1", {"subject": "grandchild", "parent_id": child_id})

    children = initialized_team.list_children("team-1", parent_id)
    assert len(children) == 1
    assert children[0]["task_id"] == child_id

    grandchildren = initialized_team.list_children("team-1", child_id)
    assert len(grandchildren) == 1
    assert grandchildren[0]["subject"] == "grandchild"

    no_children = initialized_team.list_children("team-1", "task_nope")
    assert no_children == []


# ------------------------------------------------------------------
# 32. format_task_xml without progress fields does not crash
# ------------------------------------------------------------------


def test_format_task_xml_without_progress_does_not_crash() -> None:
    """Given: TaskRecord with progress_current=None and progress_total=None.

    When: format_task_xml is called.
    Then: Returns valid XML without progress attribute, no UnboundLocalError.
    """
    record = TaskRecord(task_id="t1", subject="No progress task")

    xml = format_task_xml(record)

    assert "progress=" not in xml
    assert "No progress task" in xml
    assert "<task " in xml
    assert "</task>" in xml


def test_format_task_xml_without_progress_with_note() -> None:
    """Given: TaskRecord without progress but with last_note.

    When: format_task_xml is called.
    Then: XML includes note line, no crash.
    """
    record = TaskRecord(task_id="t1", subject="Task", last_note="Important note")

    xml = format_task_xml(record)

    assert "progress=" not in xml
    assert "note: Important note" in xml


# ------------------------------------------------------------------
# 33. concurrent register_member does not lose members
# ------------------------------------------------------------------


async def test_concurrent_register_member_preserves_all_members(
    initialized_team: FileTeamState,
) -> None:
    """Given: A team with initialized state.

    When: Multiple members are registered concurrently via asyncio.to_thread.
    Then: All members appear in state.json (no lost writes).

    This tests that register_member uses a file lock to protect the
    read-modify-write cycle on state.json.
    """
    members = [f"member_{i}" for i in range(10)]

    await asyncio.gather(
        *(
            asyncio.to_thread(initialized_team.register_member, "team-1", m, f"session-{m}")
            for m in members
        )
    )

    for m in members:
        sid = initialized_team.get_member_session_id("team-1", m)
        assert sid == f"session-{m}", f"Member {m} lost during concurrent registration"
