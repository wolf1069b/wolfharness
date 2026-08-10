"""Tests for FileDiff model alignment with OpenCode v1.4.0+ SnapshotFileDiff schema."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from wolfharness_server.opencode_server.models.common import FileDiff


pytestmark = pytest.mark.integration


# Minimal stand-in for FileChange used by from_file_change()
@dataclass
class _StubFileChange:
    path: str
    old_content: str | None
    new_content: str | None
    operation: str

    def to_unified_diff(self) -> str:
        """Return a minimal unified diff string."""
        from wolfharness.utils.diffs import compute_unified_diff

        return compute_unified_diff(
            self.old_content or "",
            self.new_content or "",
            fromfile=f"a/{self.path}",
            tofile=f"b/{self.path}",
        )


def test_filediff_schema_has_patch_no_before_after():
    """FileDiff must serialize with file/patch/additions/deletions/status — no before/after."""
    diff = FileDiff(
        file="src/main.py",
        patch="--- a/src/main.py\n+++ b/src/main.py\n@@ -1 +1 @@\n-old\n+new\n",
        additions=1,
        deletions=1,
        status="modified",
    )
    data = diff.model_dump()
    assert "file" in data
    assert "patch" in data
    assert "additions" in data
    assert "deletions" in data
    assert "status" in data
    # Must NOT contain before/after/to/from keys
    assert "before" not in data
    assert "after" not in data
    assert "to" not in data
    assert "from" not in data


def test_filediff_schema_camelcase_serialization():
    """FileDiff camelCase serialization must not leak before/after."""
    diff = FileDiff(
        file="app.ts",
        patch="patch content",
        additions=5,
        deletions=3,
        status="added",
    )
    data = diff.model_dump(by_alias=True)
    assert "file" in data
    assert "patch" in data
    assert "before" not in data
    assert "after" not in data


def test_filediff_from_file_change_populates_patch():
    """from_file_change() must store unified diff in 'patch' field."""
    change = _StubFileChange(
        path="hello.txt",
        old_content="hello world\n",
        new_content="hello universe\n",
        operation="edit",
    )
    diff = FileDiff.from_file_change(change)
    assert diff.patch is not None
    assert len(diff.patch) > 0
    # The patch should contain unified diff markers
    assert "---" in diff.patch
    assert "+++" in diff.patch
    assert diff.file == "hello.txt"
    assert diff.status == "modified"
    # Must NOT have before/after
    assert not hasattr(diff, "before")
    assert not hasattr(diff, "after")


def test_filediff_from_file_change_create_operation():
    """from_file_change() with 'create' operation must set status='added'."""
    change = _StubFileChange(
        path="new_file.py",
        old_content=None,
        new_content="print('hello')\n",
        operation="create",
    )
    diff = FileDiff.from_file_change(change)
    assert diff.status == "added"
    assert diff.patch is not None


def test_filediff_from_file_change_delete_operation():
    """from_file_change() with 'delete' operation must set status='deleted'."""
    change = _StubFileChange(
        path="old_file.py",
        old_content="print('bye')\n",
        new_content=None,
        operation="delete",
    )
    diff = FileDiff.from_file_change(change)
    assert diff.status == "deleted"
    assert diff.patch is not None


def test_filediff_from_file_change_write_operation():
    """from_file_change() with 'write' operation must set status='modified'."""
    change = _StubFileChange(
        path="config.json",
        old_content='{"key": "old"}\n',
        new_content='{"key": "new"}\n',
        operation="write",
    )
    diff = FileDiff.from_file_change(change)
    assert diff.status == "modified"


def test_filediff_additions_deletions_count():
    """Additions and deletions must be counted from the unified diff."""
    change = _StubFileChange(
        path="test.txt",
        old_content="line1\nline2\nline3\n",
        new_content="line1\nmodified2\nline3\nadded4\n",
        operation="edit",
    )
    diff = FileDiff.from_file_change(change)
    assert diff.additions > 0
    assert diff.deletions > 0


def test_filediff_patch_default_none():
    """FileDiff constructed without patch must default to None."""
    diff = FileDiff(file="empty.txt", additions=0, deletions=0)
    assert diff.patch is None


def test_filediff_status_optional():
    """FileDiff status must be optional (defaults to None)."""
    diff = FileDiff(file="test.py", patch="some patch", additions=1, deletions=0)
    assert diff.status is None


def test_filediff_status_literal_values():
    """FileDiff status must accept only 'added', 'deleted', 'modified'."""
    for status_val in ("added", "deleted", "modified"):
        diff = FileDiff(file="f.py", patch="p", additions=0, deletions=0, status=status_val)
        assert diff.status == status_val
