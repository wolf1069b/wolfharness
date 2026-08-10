"""File-based team state persistence for dynamic team mode.

Provides :class:`FileTeamState`, a synchronous file-I/O backend that stores
team metadata, member inboxes, task boards, and a versioned blackboard on
the local filesystem. All writes are atomic (tmp + ``os.replace``) and
blackboard writes are protected by :class:`filelock.FileLock` with
optimistic version locking.

Directory layout::

    {base_dir}/teams/{team_id}/
        state.json
        inboxes/{member_name}/
        tasks/
        blackboard/
        blackboard/.locks/
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import datetime
import json
from pathlib import Path
import re
import shutil
from typing import Any
import uuid

from filelock import FileLock


__all__ = [
    "FileTeamState",
    "TaskRecord",
    "format_owner_summary",
    "format_task_xml",
    "start_team_cleanup_task",
]

_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9_/]+$")

_MAX_TASKS = 100


@dataclasses.dataclass(frozen=True, slots=True)
class TaskRecord:
    """Typed representation of a task on the file-based task board.

    Mirrors the fields stored in ``{team_id}/tasks/{task_id}.json``.
    Use :meth:`from_dict` to construct from a raw JSON dict.
    """

    task_id: str
    subject: str
    description: str = ""
    owner: str = ""
    status: str = "pending"
    blocked_by: list[str] = dataclasses.field(default_factory=list)
    parent_id: str | None = None
    children: list[str] = dataclasses.field(default_factory=list)
    is_unblocked: bool = True
    last_note: str = ""
    progress_current: int | None = None
    progress_total: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskRecord:
        """Build a :class:`TaskRecord` from a raw task JSON dict."""
        return cls(
            task_id=data.get("task_id", ""),
            subject=data.get("subject", ""),
            description=data.get("description", ""),
            owner=data.get("owner", ""),
            status=data.get("status", "pending"),
            blocked_by=data.get("blocked_by", []),
            parent_id=data.get("parent_id"),
            children=data.get("children", []),
            is_unblocked=data.get("is_unblocked", True),
            last_note=data.get("last_note", ""),
            progress_current=data.get("progress_current"),
            progress_total=data.get("progress_total"),
        )


def format_task_xml(
    task: TaskRecord,
    *,
    indent: int = 2,
) -> str:
    """Format a :class:`TaskRecord` as an XML element.

    The ``owner`` attribute is always present (``owner=""`` for
    unassigned tasks).  When both ``progress_current`` and
    ``progress_total`` are set, a ``progress="{current}/{total}"``
    attribute is included.

    Args:
        task: Task record to format.
        indent: Number of spaces for indentation.

    Returns:
        XML string for the task.
    """
    pad = " " * indent
    blocked_attr = "" if task.is_unblocked else ' blocked="true"'
    progress_attr = ""
    if task.progress_current is not None and task.progress_total is not None:
        progress_attr = f' progress="{task.progress_current}/{task.progress_total}"'
    parts: list[str] = [
        (
            f'{pad}<task id="{task.task_id}" status="{task.status}" '
            f'owner="{task.owner}"{blocked_attr}{progress_attr}>'
        )
    ]
    content_line = f"{task.subject}: {task.description}" if task.description else task.subject
    parts.append(f"{pad}  {content_line}")
    if task.last_note:
        parts.append(f"{pad}  note: {task.last_note}")
    parts.append(f"{pad}</task>")
    return "\n".join(parts)


def format_owner_summary(tasks: list[TaskRecord]) -> str:
    """Return a one-line summary of task ownership distribution.

    Example::

        "4 tasks: researcher=2, analyst=1, unassigned=1"
    """
    if not tasks:
        return "0 tasks"
    counts: dict[str, int] = {}
    for t in tasks:
        owner = t.owner or "unassigned"
        counts[owner] = counts.get(owner, 0) + 1
    parts = [f"{owner}={count}" for owner, count in counts.items()]
    total = len(tasks)
    return f"{total} tasks: {', '.join(parts)}"


class FileTeamState:
    """Synchronous file-based store for team state.

    All methods perform blocking file I/O — do not call from async
    hot paths without offloading to a thread executor.
    """

    def __init__(self, base_dir: str) -> None:
        """Store the base directory for team state files.

        Args:
            base_dir: Root directory under which ``teams/`` is created.
        """
        self._base_dir = Path(base_dir)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _teams_dir(self) -> Path:
        return self._base_dir / "teams"

    def _team_dir(self, team_id: str) -> Path:
        return self._teams_dir() / team_id

    def _state_path(self, team_id: str) -> Path:
        return self._team_dir(team_id) / "state.json"

    def _inbox_dir(self, team_id: str, member_name: str) -> Path:
        return self._team_dir(team_id) / "inboxes" / member_name

    def _tasks_dir(self, team_id: str) -> Path:
        return self._team_dir(team_id) / "tasks"

    def _blackboard_dir(self, team_id: str) -> Path:
        return self._team_dir(team_id) / "blackboard"

    def _locks_dir(self, team_id: str) -> Path:
        return self._blackboard_dir(team_id) / ".locks"

    def _state_lock(self, team_id: str) -> FileLock:
        """Return a file lock protecting state.json read-modify-write cycles."""
        self._locks_dir(team_id).mkdir(parents=True, exist_ok=True)
        return FileLock(str(self._locks_dir(team_id) / "state.lock"))

    # ------------------------------------------------------------------
    # Atomic write helper
    # ------------------------------------------------------------------

    @staticmethod
    def _atomic_write(path: Path, data: dict[str, Any]) -> None:
        """Write *data* as JSON to *path* atomically.

        Writes to a sibling temporary file first, then ``os.replace``.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        """Read and parse a JSON file."""
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data

    # ------------------------------------------------------------------
    # Team lifecycle
    # ------------------------------------------------------------------

    def init(
        self,
        team_id: str,
        team_name: str,
        members: list[dict[str, str]],
    ) -> None:
        """Create the team directory structure and initial state.json.

        Args:
            team_id: Unique team identifier (used as directory name).
            team_name: Human-readable team name.
            members: List of member dicts, each with at least ``name``.
        """
        team_dir = self._team_dir(team_id)
        team_dir.mkdir(parents=True, exist_ok=True)
        self._inbox_dir(team_id, "_").parent.mkdir(parents=True, exist_ok=True)
        self._tasks_dir(team_id).mkdir(parents=True, exist_ok=True)
        self._blackboard_dir(team_id).mkdir(parents=True, exist_ok=True)
        self._locks_dir(team_id).mkdir(parents=True, exist_ok=True)

        members_map: dict[str, dict[str, str]] = {}
        for member in members:
            name = member["name"]
            members_map[name] = {
                "agent": member.get("agent", name),
                "session_id": "",
            }

        state: dict[str, Any] = {
            "team_name": team_name,
            "members": members_map,
            "status": "active",
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "ended_at": None,
        }
        self._atomic_write(self._state_path(team_id), state)

    def register_member(
        self,
        team_id: str,
        member_name: str,
        session_id: str,
        *,
        agent: str | None = None,
    ) -> None:
        """Write a member's session_id into state.json.

        Args:
            team_id: Team to update.
            member_name: Member whose session to record.
            session_id: Session identifier to persist.
            agent: Agent type (e.g. "historian").  If provided, stored
                in the member record.  If not provided, defaults to
                ``member_name`` for backward compatibility.
        """
        with self._state_lock(team_id):
            state = self._read_json(self._state_path(team_id))
            members = state["members"]
            if member_name not in members:
                members[member_name] = {
                    "agent": agent or member_name,
                    "session_id": "",
                }
            members[member_name]["session_id"] = session_id
            if agent is not None:
                members[member_name]["agent"] = agent
            self._atomic_write(self._state_path(team_id), state)

    def get_member_session_id(self, team_id: str, member_name: str) -> str | None:
        """Return the session_id for a member, or ``None`` if not registered.

        Args:
            team_id: Team to query.
            member_name: Member to look up.
        """
        state = self._read_json(self._state_path(team_id))
        members: dict[str, dict[str, str]] = state["members"]
        member = members.get(member_name)
        if member is None:
            return None
        sid: str = member.get("session_id", "")
        return sid if sid else None

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def write_message(
        self,
        team_id: str,
        member_name: str,
        message: dict[str, Any],
    ) -> None:
        """Atomically write a message to a member's inbox.

        Args:
            team_id: Team whose inbox to write to.
            member_name: Recipient member name.
            message: Message payload dict.
        """
        inbox = self._inbox_dir(team_id, member_name)
        inbox.mkdir(parents=True, exist_ok=True)
        msg_id = str(uuid.uuid4())
        path = inbox / f"{msg_id}.json"
        self._atomic_write(path, message)

    def read_messages(self, team_id: str, member_name: str) -> list[dict[str, Any]]:
        """Return all messages in a member's inbox, sorted by timestamp.

        Args:
            team_id: Team whose inbox to read.
            member_name: Member whose messages to retrieve.
        """
        inbox = self._inbox_dir(team_id, member_name)
        if not inbox.exists():
            return []
        messages = [self._read_json(f) for f in inbox.glob("*.json")]
        messages.sort(key=lambda m: m.get("timestamp", ""))
        return messages

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    def create_task(self, team_id: str, task: dict[str, Any]) -> str:
        """Create a new task file and return its task_id.

        If ``task`` contains ``parent_id``, validates that the parent
        task exists in the same team.

        Args:
            team_id: Team to add the task to.
            task: Task payload dict. May contain ``parent_id`` for
                subtask nesting.

        Raises:
            ValueError: If ``parent_id`` is set but the parent task
                does not exist.
        """
        tasks_dir = self._tasks_dir(team_id)
        tasks_dir.mkdir(parents=True, exist_ok=True)
        parent_id: str | None = task.get("parent_id")
        if parent_id is not None:
            parent_path = tasks_dir / f"{parent_id}.json"
            if not parent_path.exists():
                msg = f"Parent task not found: {parent_id}"
                raise ValueError(msg)
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        task_data = {**task, "task_id": task_id}
        if "status" not in task_data:
            task_data["status"] = "pending"
        if "blocked_by" not in task_data:
            task_data["blocked_by"] = []
        self._atomic_write(tasks_dir / f"{task_id}.json", task_data)
        return task_id

    def create_tasks_batch(  # noqa: PLR0915
        self,
        team_id: str,
        tasks: list[dict[str, Any]],
    ) -> list[str]:
        """Create multiple tasks atomically with reference resolution.

        Each task dict may contain:

        - ``subject`` (required): Short task title.
        - ``description``: Optional longer description.
        - ``owner``: Optional team member name.
        - ``blocked_by``: List of task IDs, ``#N`` positional refs,
          or symbolic ``id`` refs.
        - ``parent_id``: Parent task ID, ``#N`` ref, or symbolic ref.
        - ``id``: Symbolic name usable by other tasks in the batch.
        - ``progress_total``: Optional total for progress tracking.

        ``#N`` references resolve to the Nth task in the batch
        (0-indexed).  Symbolic references resolve by matching the
        ``id`` field of another task in the batch.

        Args:
            team_id: Team to add tasks to.
            tasks: List of task payload dicts.

        Raises:
            ValueError: If any validation fails.  No tasks are created
                on failure.

        Returns:
            List of created task IDs in batch order.
        """
        if not tasks:
            return []

        tasks_dir = self._tasks_dir(team_id)
        tasks_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Validation pass
        # ------------------------------------------------------------------
        errors: list[str] = []
        symbolic_ids: dict[str, int] = {}

        for i, task in enumerate(tasks):
            if not task.get("subject"):
                errors.append(f"Task at index {i} is missing required 'subject'")

            sym_id: str | None = task.get("id")
            if sym_id is not None:
                if sym_id in symbolic_ids:
                    errors.append(
                        f"Duplicate symbolic id '{sym_id}' at index {i} "
                        f"(first defined at index {symbolic_ids[sym_id]})"
                    )
                else:
                    symbolic_ids[sym_id] = i

        existing_count = len(list(tasks_dir.glob("*.json")))
        if existing_count + len(tasks) > _MAX_TASKS:
            errors.append(
                f"Batch would exceed max tasks limit: "
                f"{existing_count} existing + {len(tasks)} new > {_MAX_TASKS}"
            )

        # Validate #N and symbolic references in blocked_by and parent_id.
        for i, task in enumerate(tasks):
            blocked_by: list[str] = task.get("blocked_by", [])
            for ref in blocked_by:
                ref_error = self._validate_batch_ref(ref, i, len(tasks), symbolic_ids, "blocked_by")
                if ref_error is not None:
                    errors.append(f"Task at index {i}: {ref_error}")

            parent_ref: str | None = task.get("parent_id")
            if parent_ref is not None:
                ref_error = self._validate_batch_ref(
                    parent_ref, i, len(tasks), symbolic_ids, "parent_id"
                )
                if ref_error is not None:
                    errors.append(f"Task at index {i}: {ref_error}")

        if errors:
            msg = "; ".join(errors)
            raise ValueError(msg)

        # ------------------------------------------------------------------
        # Resolution pass: pre-generate IDs, build resolution map
        # ------------------------------------------------------------------
        task_ids = [f"task_{uuid.uuid4().hex[:8]}" for _ in tasks]

        # Map: #N -> task_ids[N], symbolic_id -> task_ids[index]
        ref_map: dict[str, str] = {}
        for i, tid in enumerate(task_ids):
            ref_map[f"#{i}"] = tid
        for sym_id, idx in symbolic_ids.items():
            ref_map[sym_id] = task_ids[idx]

        def resolve_ref(ref: str) -> str:
            """Resolve a reference to a real task ID.

            Returns the reference unchanged if it is neither a ``#N``
            nor a known symbolic id (i.e. it references an existing
            task outside the batch).
            """
            return ref_map.get(ref, ref)

        # ------------------------------------------------------------------
        # Creation pass: write all task files within a FileLock
        # ------------------------------------------------------------------
        lock_path = tasks_dir / ".batch.lock"
        lock = FileLock(str(lock_path))
        with lock:
            for i, (task, tid) in enumerate(zip(tasks, task_ids, strict=True)):
                resolved_blocked_by = [resolve_ref(ref) for ref in task.get("blocked_by", [])]
                resolved_parent = resolve_ref(task["parent_id"]) if task.get("parent_id") else None

                # Validate parent exists (for refs to existing tasks).
                if (
                    resolved_parent is not None
                    and not (tasks_dir / f"{resolved_parent}.json").exists()
                    and resolved_parent not in task_ids[:i]
                ):
                    msg = f"Parent task not found: {resolved_parent}"
                    raise ValueError(msg)

                task_data: dict[str, Any] = {
                    "subject": task["subject"],
                    "description": task.get("description", ""),
                    "blocked_by": resolved_blocked_by,
                    "task_id": tid,
                    "status": "pending",
                }
                if resolved_parent is not None:
                    task_data["parent_id"] = resolved_parent
                if task.get("owner"):
                    task_data["owner"] = task["owner"]
                if task.get("progress_total") is not None:
                    task_data["progress_total"] = task["progress_total"]
                self._atomic_write(tasks_dir / f"{tid}.json", task_data)

        return task_ids

    @staticmethod
    def _validate_batch_ref(  # noqa: PLR0911
        ref: str,
        task_index: int,
        batch_size: int,
        symbolic_ids: dict[str, int],
        field_name: str,
    ) -> str | None:
        """Validate a single reference in a batch task.

        Returns an error message string if invalid, ``None`` if valid.
        References that are neither ``#N`` nor a known symbolic id are
        assumed to reference existing tasks (validated at creation time).
        """
        if ref.startswith("#"):
            try:
                pos = int(ref[1:])
            except ValueError:
                return f"invalid {field_name} reference '{ref}' (not a number)"
            if pos < 0 or pos >= batch_size:
                return f"{field_name} reference '{ref}' out of range (batch has {batch_size} tasks)"
            if pos == task_index:
                return f"{field_name} reference '{ref}' cannot reference itself"
            return None
        if ref in symbolic_ids:
            if symbolic_ids[ref] == task_index:
                return f"{field_name} reference '{ref}' cannot reference itself"
            return None
        # Not a #N ref or symbolic id — assumed to be an existing task ID.
        return None

    def get_task(self, team_id: str, task_id: str) -> dict[str, Any] | None:
        """Return a single task by ID, or ``None`` if not found.

        Args:
            team_id: Team containing the task.
            task_id: Task to retrieve.
        """
        path = self._tasks_dir(team_id) / f"{task_id}.json"
        if not path.exists():
            return None
        return self._read_json(path)

    def list_tasks(self, team_id: str) -> list[dict[str, Any]]:
        """Return all tasks with computed ``is_unblocked`` and ``children`` fields.

        A task is unblocked when all ``blocked_by`` tasks have
        ``status == "completed"``. Failed dependencies do NOT unblock.

        The ``children`` field is a list of task_ids that have
        ``parent_id`` equal to this task's ``task_id``.

        Args:
            team_id: Team whose tasks to list.
        """
        tasks_dir = self._tasks_dir(team_id)
        if not tasks_dir.exists():
            return []
        tasks: list[dict[str, Any]] = []
        task_by_id: dict[str, dict[str, Any]] = {}
        for f in tasks_dir.glob("*.json"):
            t = self._read_json(f)
            tasks.append(t)
            tid: str = t.get("task_id", f.stem)
            task_by_id[tid] = t

        # Compute children: for each task, find all tasks whose parent_id == this task_id.
        for t in tasks:
            t["children"] = []

        for t in tasks:
            pid: str | None = t.get("parent_id")
            if pid is not None and pid in task_by_id:
                parent = task_by_id[pid]
                tid = t.get("task_id", "")
                parent.setdefault("children", []).append(tid)

        for t in tasks:
            blocked_by: list[str] = t.get("blocked_by", [])
            if not blocked_by:
                t["is_unblocked"] = True
                continue
            deps = [task_by_id.get(dep) for dep in blocked_by]
            t["is_unblocked"] = all(
                dep is not None and dep.get("status") == "completed" for dep in deps
            )
        return tasks

    def list_children(self, team_id: str, parent_id: str) -> list[dict[str, Any]]:
        """Return tasks that are direct children of *parent_id*.

        Args:
            team_id: Team containing the tasks.
            parent_id: Parent task ID to filter by.
        """
        tasks = self.list_tasks(team_id)
        return [t for t in tasks if t.get("parent_id") == parent_id]

    def update_task(
        self,
        team_id: str,
        task_id: str,
        updates: dict[str, Any],
        *,
        progress_current: int | None = None,
        progress_total: int | None = None,
    ) -> dict[str, Any]:
        """Merge *updates* into an existing task and return the result.

        Args:
            team_id: Team containing the task.
            task_id: Task to update.
            updates: Fields to merge into the task.
            progress_current: Optional current progress value to persist.
            progress_total: Optional total progress value to persist.
        """
        path = self._tasks_dir(team_id) / f"{task_id}.json"
        task = self._read_json(path)
        task.update(updates)
        if progress_current is not None:
            task["progress_current"] = progress_current
        if progress_total is not None:
            task["progress_total"] = progress_total
        self._atomic_write(path, task)
        return task

    # ------------------------------------------------------------------
    # Blackboard
    # ------------------------------------------------------------------

    def _validate_key(self, key: str, blackboard_dir: Path) -> Path:
        """Validate a blackboard key and return the safe file path.

        Args:
            key: Blackboard key to validate.
            blackboard_dir: Resolved blackboard directory.

        Raises:
            ValueError: If the key contains invalid characters or
                attempts path traversal.
        """
        if not _KEY_PATTERN.match(key):
            msg = f"Invalid blackboard key: {key!r}"
            raise ValueError(msg)
        key_path = (blackboard_dir / f"{key}.json").resolve()
        bb_resolved = blackboard_dir.resolve()
        try:
            key_path.relative_to(bb_resolved)
        except ValueError:
            msg = f"Path traversal detected in blackboard key: {key!r}"
            raise ValueError(msg) from None
        return key_path

    def read_blackboard(self, team_id: str, key: str) -> dict[str, Any] | None:
        """Return the blackboard value + metadata for *key*, or ``None``.

        Args:
            team_id: Team whose blackboard to read.
            key: Blackboard key.
        """
        bb_dir = self._blackboard_dir(team_id)
        key_path = self._validate_key(key, bb_dir)
        if not key_path.exists():
            return None
        return self._read_json(key_path)

    def write_blackboard(
        self,
        team_id: str,
        key: str,
        value: dict[str, Any],
        expected_version: int | None = None,
        written_by: str = "unknown",
        mode: str = "overwrite",
    ) -> str:
        """Write a value to the blackboard with optimistic locking.

        Args:
            team_id: Team whose blackboard to write to.
            key: Blackboard key.
            value: Value payload to store.
            expected_version: Expected current version for optimistic
                locking.  If ``None``, no version check is performed.
            written_by: Name of the writer.
            mode: ``"overwrite"`` (default) replaces the value entirely;
                ``"append"`` concatenates to the existing ``text`` field.

        Returns:
            ``"Written, version=N"`` on success, or
            ``"Conflict: current version is N"`` on version mismatch.
        """
        bb_dir = self._blackboard_dir(team_id)
        self._locks_dir(team_id).mkdir(parents=True, exist_ok=True)
        key_path = self._validate_key(key, bb_dir)
        lock_path = self._locks_dir(team_id) / f"{key}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(lock_path))
        with lock:
            current: dict[str, Any] | None = None
            current_version = 0
            if key_path.exists():
                current = self._read_json(key_path)
                current_version = current.get("version", 0)

            if expected_version is not None and expected_version != current_version:
                return f"Conflict: current version is {current_version}"

            new_version = current_version + 1

            if mode == "append" and current is not None:
                old_text = current.get("value", {}).get("text", "")
                new_text = value.get("text", "")
                if old_text:
                    merged_value: dict[str, Any] = {"text": old_text + "\n" + new_text}
                else:
                    merged_value = {"text": new_text}
            else:
                merged_value = value

            entry: dict[str, Any] = {
                "value": merged_value,
                "version": new_version,
                "written_by": written_by,
                "written_at": datetime.datetime.now(datetime.UTC).isoformat(),
            }
            self._atomic_write(key_path, entry)
            return f"Written, version={new_version}"

    def list_blackboard(self, team_id: str) -> list[str]:
        """Return all blackboard keys (without the ``.json`` suffix).

        Args:
            team_id: Team whose blackboard to list.
        """
        bb_dir = self._blackboard_dir(team_id)
        if not bb_dir.exists():
            return []
        bb_resolved = bb_dir.resolve()
        keys = [
            str(f.relative_to(bb_resolved).with_suffix("")) for f in bb_resolved.rglob("*.json")
        ]
        return sorted(keys)

    def delete_blackboard(self, team_id: str, key: str) -> None:
        """Delete a blackboard key.

        Args:
            team_id: Team whose blackboard to modify.
            key: Blackboard key to delete.
        """
        bb_dir = self._blackboard_dir(team_id)
        key_path = self._validate_key(key, bb_dir)
        if key_path.exists():
            key_path.unlink()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self, team_id: str) -> None:
        """Remove the entire team directory.

        Args:
            team_id: Team to remove.
        """
        team_dir = self._team_dir(team_id)
        if team_dir.exists():
            shutil.rmtree(team_dir)

    @classmethod
    def cleanup_expired_teams(cls, base_dir: str, ttl_hours: int) -> int:
        """Remove expired team directories and mark orphaned teams.

        Scans ``{base_dir}/teams/`` for ``state.json`` files. Teams with
        ``status="deleted"`` and ``ended_at`` older than *ttl_hours* are
        removed entirely. Teams with ``status="active"`` whose
        ``created_at`` + *ttl_hours* < now and ``ended_at`` is ``None``
        are marked as ``status="orphaned"`` (best-effort write).

        Args:
            base_dir: Root directory containing ``teams/``.
            ttl_hours: Minimum age (in hours) before cleanup or orphaning.

        Returns:
            Number of team directories removed.
        """
        teams_root = Path(base_dir) / "teams"
        if not teams_root.exists():
            return 0
        now = datetime.datetime.now(datetime.UTC)
        removed = 0
        for entry in teams_root.iterdir():
            if not entry.is_dir():
                continue
            state_path = entry / "state.json"
            if not state_path.exists():
                continue
            state = cls._read_json(state_path)
            status: str = state.get("status", "")

            if status == "deleted":
                ended_at_raw: str | None = state.get("ended_at")
                if ended_at_raw is None:
                    continue
                try:
                    ended_at = datetime.datetime.fromisoformat(ended_at_raw)
                except ValueError:
                    continue
                if ended_at.tzinfo is None:
                    ended_at = ended_at.replace(tzinfo=datetime.UTC)
                age_hours = (now - ended_at).total_seconds() / 3600
                if age_hours >= ttl_hours:
                    shutil.rmtree(entry)
                    removed += 1

            elif status == "active":
                created_at_raw: str | None = state.get("created_at")
                if created_at_raw is None:
                    continue
                if state.get("ended_at") is not None:
                    continue
                try:
                    created_at = datetime.datetime.fromisoformat(created_at_raw)
                except ValueError:
                    continue
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=datetime.UTC)
                age_hours = (now - created_at).total_seconds() / 3600
                if age_hours >= ttl_hours:
                    state["status"] = "orphaned"
                    with contextlib.suppress(OSError):
                        cls._atomic_write(state_path, state)

        return removed


async def start_team_cleanup_task(
    base_dir: str,
    ttl_hours: int,
    interval_minutes: int = 10,
) -> asyncio.Task[None]:
    """Start a background task that periodically cleans up expired teams.

    Args:
        base_dir: Root directory containing ``teams/``.
        ttl_hours: Minimum age (in hours) before cleanup or orphaning.
        interval_minutes: Seconds between cleanup runs, expressed in minutes.

    Returns:
        The ``asyncio.Task`` for cancellation.
    """
    try:
        import logfire
    except ImportError:
        logfire = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]

    async def _cleanup_loop() -> None:
        if logfire is not None:
            span_ctx = logfire.span(
                "lifecycle.team_cleanup",
                base_dir=base_dir,
                ttl_hours=ttl_hours,
            )
        else:
            from contextlib import nullcontext

            span_ctx = nullcontext()
        with span_ctx:
            while True:
                removed = FileTeamState.cleanup_expired_teams(base_dir, ttl_hours)
                if removed > 0 and logfire is not None:
                    logfire.info(
                        "team_cleanup_removed",
                        removed=removed,
                        base_dir=base_dir,
                    )
                await asyncio.sleep(interval_minutes * 60)

    return asyncio.create_task(_cleanup_loop())
