"""Tests for OpenCodeStorageProvider project storage methods.

Covers all 7 project methods:
- save_project, get_project, get_project_by_worktree, get_project_by_name
- list_projects, delete_project, touch_project
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import tempfile

import pytest

from wolfharness.sessions.models import ProjectData
from wolfharness_config.storage import OpenCodeStorageConfig
from wolfharness_storage.opencode_provider import OpenCodeStorageProvider


pytestmark = pytest.mark.unit


@pytest.fixture
async def provider():
    """Create an OpenCode provider with temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = OpenCodeStorageConfig(path=tmpdir)
        prov = OpenCodeStorageProvider(config)
        async with prov:
            yield prov


def _make_project(
    *,
    project_id: str = "test_project_001",
    worktree: str = "/tmp/test-project",
    name: str | None = "test-project",
    vcs: str | None = "git",
    config_path: str | None = None,
    settings: dict | None = None,
    last_active: datetime | None = None,
) -> ProjectData:
    """Helper to create ProjectData instances for tests."""
    return ProjectData(
        project_id=project_id,
        worktree=worktree,
        name=name,
        vcs=vcs,
        config_path=config_path,
        settings=settings or {},
        last_active=last_active or datetime.now(UTC),
    )


async def test_save_and_get_project(provider: OpenCodeStorageProvider) -> None:
    """Test saving a project and retrieving it by ID."""
    project = _make_project(
        project_id="proj_001",
        worktree="/home/user/project-a",
        name="project-a",
        vcs="git",
        config_path="/home/user/project-a/.wolfharness.yml",
        settings={"model": "openai:gpt-4o"},
    )

    await provider.save_project(project)

    result = await provider.get_project("proj_001")

    assert result is not None
    assert result.project_id == "proj_001"
    assert result.worktree == "/home/user/project-a"
    assert result.name == "project-a"
    assert result.vcs == "git"
    assert result.config_path == "/home/user/project-a/.wolfharness.yml"
    assert result.settings == {"model": "openai:gpt-4o"}


async def test_get_project_not_found(provider: OpenCodeStorageProvider) -> None:
    """Test that getting a nonexistent project returns None."""
    result = await provider.get_project("nonexistent_id")
    assert result is None


async def test_get_project_by_worktree(provider: OpenCodeStorageProvider) -> None:
    """Test finding a project by worktree path with path resolution."""
    worktree = str(Path("/tmp/test-worktree-project").resolve())
    project = _make_project(
        project_id="proj_worktree",
        worktree=worktree,
    )
    await provider.save_project(project)

    # Search by the same resolved path
    result = await provider.get_project_by_worktree(worktree)
    assert result is not None
    assert result.project_id == "proj_worktree"

    # Search with unresolved path — should still resolve and match
    unresolved = "/tmp/test-worktree-project"
    result2 = await provider.get_project_by_worktree(unresolved)
    assert result2 is not None
    assert result2.project_id == "proj_worktree"


async def test_get_project_by_name(provider: OpenCodeStorageProvider) -> None:
    """Test finding a project by its friendly name."""
    project = _make_project(
        project_id="proj_named",
        name="my-special-project",
    )
    await provider.save_project(project)

    result = await provider.get_project_by_name("my-special-project")
    assert result is not None
    assert result.project_id == "proj_named"

    # Nonexistent name returns None
    result2 = await provider.get_project_by_name("nonexistent-name")
    assert result2 is None


async def test_list_projects_sorted_by_last_active(provider: OpenCodeStorageProvider) -> None:
    """Test that list_projects returns items sorted by last_active descending."""
    project_a = _make_project(
        project_id="proj_a",
        name="alpha",
        last_active=datetime(2025, 1, 1, tzinfo=UTC),
    )
    project_b = _make_project(
        project_id="proj_b",
        name="beta",
        last_active=datetime(2025, 6, 15, tzinfo=UTC),
    )
    project_c = _make_project(
        project_id="proj_c",
        name="gamma",
        last_active=datetime(2025, 3, 10, tzinfo=UTC),
    )

    await provider.save_project(project_a)
    await provider.save_project(project_b)
    await provider.save_project(project_c)

    result = await provider.list_projects()

    assert len(result) == 3
    # Sorted by last_active descending: beta (June), gamma (March), alpha (Jan)
    assert result[0].project_id == "proj_b"
    assert result[1].project_id == "proj_c"
    assert result[2].project_id == "proj_a"


async def test_list_projects_with_limit(provider: OpenCodeStorageProvider) -> None:
    """Test that the limit parameter works correctly."""
    project_a = _make_project(
        project_id="proj_limit_a",
        last_active=datetime(2025, 1, 1, tzinfo=UTC),
    )
    project_b = _make_project(
        project_id="proj_limit_b",
        last_active=datetime(2025, 6, 15, tzinfo=UTC),
    )
    project_c = _make_project(
        project_id="proj_limit_c",
        last_active=datetime(2025, 3, 10, tzinfo=UTC),
    )

    await provider.save_project(project_a)
    await provider.save_project(project_b)
    await provider.save_project(project_c)

    result = await provider.list_projects(limit=2)
    assert len(result) == 2
    # Should be the two most recently active
    assert result[0].project_id == "proj_limit_b"
    assert result[1].project_id == "proj_limit_c"


async def test_delete_project(provider: OpenCodeStorageProvider) -> None:
    """Test deleting a project removes the file and returns True."""
    project = _make_project(project_id="proj_delete_me")
    await provider.save_project(project)

    # Verify it exists
    result = await provider.get_project("proj_delete_me")
    assert result is not None

    # Delete it
    deleted = await provider.delete_project("proj_delete_me")
    assert deleted is True

    # Verify it's gone
    result2 = await provider.get_project("proj_delete_me")
    assert result2 is None

    # JSON file should be removed
    project_file = provider.projects_path / "proj_delete_me.json"
    assert not project_file.exists()


async def test_delete_project_not_found(provider: OpenCodeStorageProvider) -> None:
    """Test deleting a nonexistent project returns False."""
    deleted = await provider.delete_project("nonexistent_project")
    assert deleted is False


async def test_touch_project_updates_timestamp(provider: OpenCodeStorageProvider) -> None:
    """Test that touch_project updates the last_active timestamp."""
    original_time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=UTC)
    project = _make_project(
        project_id="proj_touch",
        last_active=original_time,
    )
    await provider.save_project(project)

    # Verify original timestamp
    result = await provider.get_project("proj_touch")
    assert result is not None
    assert result.last_active.year == 2020

    # Touch it
    await provider.touch_project("proj_touch")

    # Verify timestamp was updated
    result2 = await provider.get_project("proj_touch")
    assert result2 is not None
    assert result2.last_active > original_time


async def test_list_projects_handles_corrupted_file(provider: OpenCodeStorageProvider) -> None:
    """Test that corrupted JSON files are skipped without failing the whole listing."""
    project = _make_project(
        project_id="proj_good",
        name="good-project",
    )
    await provider.save_project(project)

    # Write a corrupted file directly
    corrupted_file = provider.projects_path / "proj_bad.json"
    corrupted_file.write_text("{ this is not valid json }", encoding="utf-8")

    result = await provider.list_projects()

    # Should still return the good project
    assert len(result) == 1
    assert result[0].project_id == "proj_good"
