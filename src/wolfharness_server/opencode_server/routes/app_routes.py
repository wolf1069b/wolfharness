"""App, project, path, and VCS routes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import anyenv
from fastapi import APIRouter, HTTPException

from wolfharness.utils.time_utils import datetime_to_ms
from wolfharness_server.opencode_server.dependencies import StateDep
from wolfharness_server.opencode_server.models import (
    App,
    AppTimeInfo,
    PathInfo,
    Project,
    ProjectDirectory,
    ProjectTime,
    ProjectUpdatedEvent,
    ProjectUpdateRequest,
    VcsInfo,
)
from wolfharness_storage.opencode_provider import helpers
from wolfharness_storage.project_store import ProjectStore


if TYPE_CHECKING:
    from wolfharness.sessions.models import ProjectData


router = APIRouter(tags=["app"])


@router.get("/app")
async def get_app(state: StateDep) -> App:
    """Get app information."""
    working_path = Path(state.working_dir)
    is_git = (working_path / ".git").is_dir()
    worktree = await _find_worktree(state.working_dir)
    path_info = PathInfo.for_directory(state.working_dir, worktree=worktree)
    time_info = AppTimeInfo(initialized=state.start_time)
    return App(git=is_git, hostname="localhost", path=path_info, time=time_info)


def _project_data_to_response(data: ProjectData, *, project_id: str | None = None) -> Project:
    """Convert ProjectData to OpenCode Project response.

    Args:
        data: Internal ProjectData from storage.
        project_id: Optional override for the project ID. When provided, uses
            the OpenCode-compatible project ID (git root commit SHA1) instead
            of the internal path-based ID. This is required for TUI event
            routing compatibility after OpenCode v1.4.4+.
    """
    working_path = Path(data.worktree)
    match data.vcs:
        case "git":
            vcs_dir: str | None = str(working_path / ".git")
        case "hg":
            vcs_dir = str(working_path / ".hg")
        case _:
            vcs_dir = None
    return Project(
        id=project_id or data.project_id,
        worktree=data.worktree,
        vcs_dir=vcs_dir,
        vcs=data.vcs,
        time=ProjectTime(created=datetime_to_ms(data.created_at)),
    )


async def _get_current_project(state: StateDep) -> ProjectData:
    """Get or create the current project from storage.

    The returned ``ProjectData`` carries ``project_id`` from
    ``generate_project_id()`` (a SHA1 of the worktree path — AgentPool's
    internal identifier). That internal project ID is distinct from OpenCode's
    git-root-based session layout ID; the two IDs are not interchangeable.
    """
    project_store = ProjectStore(state.storage)
    return await project_store.get_or_create(state.working_dir)


@router.get("/project")
async def list_projects(state: StateDep) -> list[Project]:
    """List all projects."""
    project_store = ProjectStore(state.storage)
    projects = await project_store.list_recent(limit=50)
    return [_project_data_to_response(p) for p in projects]


@router.get("/project/current")
async def get_project_current(state: StateDep) -> Project:
    """Get current project.

    Returns the OpenCode-compatible project ID (git root commit SHA1) so
    that the TUI's event routing filter (event.project === project.project())
    matches the project field in GlobalEvent SSE envelopes.
    """
    project = await _get_current_project(state)
    opencode_project_id = helpers.compute_project_id(project.worktree)
    return _project_data_to_response(project, project_id=opencode_project_id)


@router.patch("/project/{project_id}")
async def update_project(project_id: str, update: ProjectUpdateRequest, state: StateDep) -> Project:
    """Update project metadata (name, settings).

    Emits a project.updated event when successful.

    Args:
        project_id: Project identifier
        update: Fields to update (name and/or settings)
        state: Server state

    Returns:
        Updated project data

    Raises:
        HTTPException: If project not found
    """
    store = ProjectStore(state.storage)
    project_data = None
    # Update name if provided
    if update.name is not None:
        project_data = await store.set_name(project_id, update.name)
        if not project_data:
            raise HTTPException(status_code=404, detail="Project not found")
    # Update settings if provided
    if update.settings:
        project_data = await store.update_settings(project_id, **update.settings)
        if not project_data:
            raise HTTPException(status_code=404, detail="Project not found")
    # If neither name nor settings provided, just fetch the project
    if not project_data:
        project_data = await store.get_by_id(project_id)
        if not project_data:
            raise HTTPException(status_code=404, detail="Project not found")

    # Convert to OpenCode Project model
    project = _project_data_to_response(project_data)
    # Broadcast event
    await state.broadcast_event(ProjectUpdatedEvent.create(project))
    return project


@router.get("/project/{project_id}/directories")
async def list_project_directories(project_id: str, state: StateDep) -> list[ProjectDirectory]:
    """List directories for a project.

    Returns the known directories for the project.  For AgentPool
    (single-directory mode) this always returns a single 'main'
    directory pointing to the server's working directory.

    This endpoint is required by the OpenCode TUI for event routing
    and file operation baselines.

    Per the OpenCode API contract, this returns a plain array (not
    wrapped in a ``data`` object).  The SDK layers add the wrapper.

    Args:
        project_id: Project identifier (ignored; AgentPool is single-project).
        state: Server state (injected dependency).

    Returns:
        List of directory entries.
    """
    return [
        ProjectDirectory(
            directory=state.working_dir,
            type="main",
        ),
    ]


@router.get("/path")
async def get_path(state: StateDep) -> PathInfo:
    """Get current path info."""
    worktree = await _find_worktree(state.working_dir)
    return PathInfo.for_directory(state.working_dir, worktree=worktree)


async def _find_worktree(directory: str) -> str | None:
    """Find the git worktree root for the given directory."""
    return await _run_command("git", ["rev-parse", "--show-toplevel"], directory)


async def _run_command(cmd: str, args: list[str], cwd: str) -> str | None:
    """Run a git command asynchronously and return stdout, or None on error."""
    try:
        proc = await anyenv.create_process(cmd, *args, cwd=cwd, stdout="pipe", stderr="pipe")
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return None
        return stdout.decode().strip()
    except OSError:
        return None


@router.get("/vcs")
async def get_vcs(state: StateDep) -> VcsInfo:
    """Get VCS info.

    TODO: For remote/ACP support, these git commands should run through
    state.env.execute_command() instead of subprocess.run() so they
    execute on the client side where the repository lives.
    """
    git_dir = Path(state.working_dir) / ".git"
    if not git_dir.is_dir():
        return VcsInfo()

    branch, commit, status = await asyncio.gather(
        _run_command("git", ["rev-parse", "--abbrev-ref", "HEAD"], state.working_dir),
        _run_command("git", ["rev-parse", "HEAD"], state.working_dir),
        _run_command("git", ["status", "--porcelain"], state.working_dir),
    )

    return VcsInfo(branch=branch, dirty=bool(status), commit=commit)
