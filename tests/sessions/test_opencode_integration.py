"""Integration tests for OpenCode storage persistence and dual-ID scheme.

Covers:
- Project and session persistence across provider restart
- Accessibility of both project ID schemes (compute_project_id / generate_project_id)
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

import pytest

from wolfharness.sessions.models import ProjectData, SessionData
from wolfharness.utils.identifiers import ascending
from wolfharness_config.storage import OpenCodeStorageConfig
from wolfharness_storage.opencode_provider import OpenCodeStorageProvider
from wolfharness_storage.opencode_provider.helpers import compute_project_id
from wolfharness_storage.project_store import generate_project_id


pytestmark = pytest.mark.integration


@pytest.fixture
async def provider():
    """Create an OpenCode provider with temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = OpenCodeStorageConfig(path=tmpdir)
        prov = OpenCodeStorageProvider(config)
        async with prov:
            yield prov


def _init_git_repo(directory: str) -> None:
    """Initialize a minimal git repo so compute_project_id returns a commit SHA."""
    subprocess.run(["git", "init"], cwd=directory, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=directory,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=directory,
        capture_output=True,
        check=True,
    )
    # Create an initial commit so there's a root commit SHA
    dummy = Path(directory) / "README.md"
    dummy.write_text("init", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=directory, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=directory,
        capture_output=True,
        check=True,
    )


async def test_project_session_persistence_across_restart(
    provider: OpenCodeStorageProvider,
) -> None:
    """Project and sessions survive closing and re-opening the provider."""
    base_path = provider.base_path

    # Create project data
    worktree = str(Path(base_path) / "my_project")
    Path(worktree).mkdir(parents=True, exist_ok=True)
    _init_git_repo(worktree)

    project_id = generate_project_id(worktree)
    project = ProjectData(
        project_id=project_id,
        worktree=worktree,
        name="test-project",
        vcs="git",
    )
    await provider.save_project(project)

    # Create session data for this project
    session_id = ascending("session")
    session = SessionData(
        session_id=session_id,
        agent_name="test-agent",
        project_id=project_id,
        cwd=worktree,
    )
    await provider.save_session(session)

    # Verify within the same provider instance
    loaded_project = await provider.get_project(project_id)
    assert loaded_project is not None
    assert loaded_project.project_id == project_id
    assert loaded_project.worktree == worktree
    assert loaded_project.name == "test-project"

    loaded_sessions = await provider.list_session_ids()
    assert session_id in loaded_sessions

    # Close the provider (simulate restart)
    await provider.__aexit__(None, None, None)

    # Create a NEW provider instance with the same base path
    config = OpenCodeStorageConfig(path=str(base_path))
    new_provider = OpenCodeStorageProvider(config)
    async with new_provider:
        # Verify project is recovered
        recovered_project = await new_provider.get_project(project_id)
        assert recovered_project is not None
        assert recovered_project.project_id == project_id
        assert recovered_project.worktree == worktree
        assert recovered_project.name == "test-project"

        # Verify sessions persist
        recovered_sessions = await new_provider.list_session_ids()
        assert session_id in recovered_sessions

        # Verify get_project_by_worktree returns the project
        by_worktree = await new_provider.get_project_by_worktree(worktree)
        assert by_worktree is not None
        assert by_worktree.project_id == project_id


async def test_both_project_ids_accessible() -> None:
    """Both compute_project_id and generate_project_id return valid values.

    The two functions use different algorithms and serve different purposes:
    - compute_project_id: git root commit SHA1 (OpenCode session layout)
    - generate_project_id: path SHA1 (AgentPool project registry)

    This test documents that both are accessible and produce distinct values
    for the same directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir) / "dual_id_project"
        project_dir.mkdir(parents=True, exist_ok=True)
        _init_git_repo(str(project_dir))

        opencode_id = compute_project_id(str(project_dir))
        wolfharness_id = generate_project_id(str(project_dir))

        # Both should return valid hex strings (or "global" for compute_project_id)
        assert len(opencode_id) >= 1
        assert len(wolfharness_id) == 40  # SHA1 hex digest

        # With a git repo, compute_project_id returns the root commit SHA
        assert opencode_id != "global"
        assert len(opencode_id) == 40  # commit SHA1 hex digest

        # They are different algorithms — expect different values
        assert opencode_id != wolfharness_id, (
            "compute_project_id (git root SHA) and generate_project_id (path SHA) "
            "use different algorithms and should produce different values"
        )


async def test_compute_project_id_without_git() -> None:
    """compute_project_id returns 'global' when not in a git repo."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # No git init — should return "global"
        result = compute_project_id(tmpdir)
        assert result == "global"


async def test_generate_project_id_deterministic() -> None:
    """generate_project_id returns the same value for the same path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        id_1 = generate_project_id(tmpdir)
        id_2 = generate_project_id(tmpdir)
        assert id_1 == id_2
        assert len(id_1) == 40
