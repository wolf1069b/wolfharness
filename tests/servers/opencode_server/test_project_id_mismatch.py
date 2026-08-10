"""Red flag test: project ID mismatch between /project/current and GlobalEvent envelope.

This test demonstrates the bug where the OpenCode TUI client drops all events
because the project ID from /project/current does not match the project field
in GlobalEvent SSE envelopes.

Root cause:
- /project/current returns generate_project_id() = SHA1 of worktree path
- GlobalEvent uses compute_project_id() = git root commit SHA1

After OpenCode commit 2b432d9e03 (fix(tui): scope events by project), the TUI
filters events by: event.project === project.project()

When these IDs differ, ALL events are silently dropped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from wolfharness_server.opencode_server.dependencies import get_state
from wolfharness_server.opencode_server.models import SessionStatusEvent
from wolfharness_server.opencode_server.routes import app_router, global_router
from wolfharness_server.opencode_server.routes.global_routes import GlobalEventFactory
from wolfharness_storage.opencode_provider.helpers import compute_project_id


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from wolfharness_server.opencode_server.state import ServerState


@pytest.fixture
def app_with_app_routes(server_state: ServerState) -> FastAPI:
    """Create a FastAPI app with app and global routes for testing."""
    app = FastAPI()
    app.include_router(app_router)
    app.include_router(global_router)
    app.dependency_overrides[get_state] = lambda: server_state
    return app


@pytest.fixture
async def async_client_with_app_routes(app_with_app_routes: FastAPI) -> AsyncIterator[AsyncClient]:
    """Create an async test client with app routes."""
    transport = ASGITransport(app=app_with_app_routes)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.integration
@pytest.mark.anyio
async def test_project_current_id_matches_global_event_project(
    async_client_with_app_routes: AsyncClient,
    server_state: ServerState,
    tmp_git_dir: Path,
) -> None:
    """RED FLAG: /project/current id must match GlobalEvent project field.

    This test proves the bug: the project ID returned by /project/current
    uses generate_project_id() (path SHA1), while GlobalEvent envelopes use
    compute_project_id() (git root commit SHA1). After OpenCode v1.4.4+ TUI
    scopes events by project, this mismatch causes all events to be dropped.
    """
    # Update server state to use the git directory
    server_state.working_dir = str(tmp_git_dir)
    server_state._event_factory = None  # Force recreation with new working_dir

    # 1. Get project ID from /project/current
    response = await async_client_with_app_routes.get("/project/current")
    assert response.status_code == 200
    project_data: dict[str, Any] = response.json()
    project_current_id = project_data["id"]

    # 2. Compute what GlobalEventFactory will use for project
    expected_global_event_project = compute_project_id(str(tmp_git_dir))

    # 3. Verify /project/current now returns the OpenCode-compatible ID
    assert project_current_id == expected_global_event_project, (
        "Expected /project/current to return OpenCode-compatible git-commit-based ID"
    )

    # 4. Create a GlobalEvent and check its project field
    factory = GlobalEventFactory(
        directory=str(tmp_git_dir.resolve()),
        project=expected_global_event_project,
    )
    event = SessionStatusEvent.create(session_id="s1", status_type="busy")
    envelope_json = factory.wrap(event)
    import json

    envelope = json.loads(envelope_json)
    global_event_project = envelope["project"]

    # 5. THE RED FLAG: these MUST match for TUI event routing to work
    # This assertion FAILS with the current code, proving the bug.
    assert project_current_id == global_event_project, (
        f"CRITICAL MISMATCH: /project/current returns id={project_current_id!r} "
        f"but GlobalEvent uses project={global_event_project!r}. "
        f"OpenCode TUI will drop all events. "
        f"Fix: make /project/current return compute_project_id() instead."
    )
