"""L4 subprocess E2E tests for OpenCode shell, diff, todo, health, and init endpoints.

Covers Phase C groups C9 (3 tasks) and C10 (2 tasks):
    - C9.1 test_post_shell: POST /session/{id}/shell → 200
    - C9.2 test_get_diff: GET /session/{id}/diff → 200
    - C9.3 test_get_todo: GET /session/{id}/todo → 200
    - C10.1 test_get_health: GET /global/health → 200 with health status
    - C10.2 test_post_init: POST /session/{id}/init → 200

All tests use ``model: test`` (pydantic-ai TestModel) so NO API key is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import pytest

from tests.e2e.conftest import SKIP_NO_BINARY, SKIP_WINDOWS


if TYPE_CHECKING:
    from tests.e2e.conftest import SubprocessServer


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(SKIP_NO_BINARY, reason="wolfharness binary not on PATH"),
    pytest.mark.skipif(SKIP_WINDOWS, reason="Windows subprocess issues"),
]

# Shared parametrize for the subprocess_server fixture.
_OPENCODE_PARAMS: dict[str, Any] = {
    "serve_command": "serve-opencode",
    "is_stdio": False,
    "health_path": "/session",
}


async def _create_session(client: httpx.AsyncClient, base_url: str) -> str:
    """Create a session and return its ID."""
    resp = await client.post(f"{base_url}/session", json={})
    assert resp.status_code in (200, 201), (
        f"Failed to create session: {resp.status_code} {resp.text}"
    )
    session_data = resp.json()
    session_id = session_data.get("id") or session_data.get("sessionID")
    assert session_id, f"Expected session ID in response: {session_data}"
    return session_id


# ---------------------------------------------------------------------------
# C9.1 — POST /session/{session_id}/shell
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subprocess_server", [_OPENCODE_PARAMS], indirect=True)
async def test_post_shell(subprocess_server: SubprocessServer) -> None:
    """C9.1: POST /session/{session_id}/shell, verify 200 with command output."""
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=30.0) as client:
        session_id = await _create_session(client, base_url)

        # POST shell command — ShellRequest requires `agent` and `command`.
        shell_body: dict[str, Any] = {
            "agent": "test_agent",
            "command": "echo hello",
        }
        resp = await client.post(
            f"{base_url}/session/{session_id}/shell",
            json=shell_body,
        )
        assert resp.status_code == 200, (
            f"Expected 200 for POST /shell, got {resp.status_code}: {resp.text}"
        )
        # Response is a MessageWithParts with the shell output in a text part.
        result = resp.json()
        assert "info" in result, f"Expected 'info' in shell response: {result}"
        assert "parts" in result, f"Expected 'parts' in shell response: {result}"
        # Verify the shell output appears in one of the text parts.
        parts = result["parts"]
        text_parts = [p for p in parts if p.get("type") == "text"]
        assert len(text_parts) >= 1, f"Expected at least one text part in shell response: {parts}"
        # The text part should contain "echo hello" and "hello" in the output.
        shell_output = text_parts[-1].get("text", "")
        assert "hello" in shell_output, f"Expected 'hello' in shell output, got: {shell_output}"


# ---------------------------------------------------------------------------
# C9.2 — GET /session/{session_id}/diff
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subprocess_server", [_OPENCODE_PARAMS], indirect=True)
async def test_get_diff(subprocess_server: SubprocessServer) -> None:
    """C9.2: GET /session/{session_id}/diff, verify 200 with diff list."""
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        session_id = await _create_session(client, base_url)

        # GET diff — returns a list of FileDiff objects (empty if no changes).
        resp = await client.get(f"{base_url}/session/{session_id}/diff")
        assert resp.status_code == 200, (
            f"Expected 200 for GET /diff, got {resp.status_code}: {resp.text}"
        )
        diffs = resp.json()
        assert isinstance(diffs, list), f"Expected list response, got {type(diffs)}: {diffs}"
        # With TestModel (no file edits), the diff list should be empty.
        assert len(diffs) == 0, f"Expected no file diffs with TestModel, got {diffs}"


# ---------------------------------------------------------------------------
# C9.3 — GET /session/{session_id}/todo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subprocess_server", [_OPENCODE_PARAMS], indirect=True)
async def test_get_todo(subprocess_server: SubprocessServer) -> None:
    """C9.3: GET /session/{session_id}/todo, verify 200 with todo list."""
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        session_id = await _create_session(client, base_url)

        # GET todo — returns a list of Todo objects.
        resp = await client.get(f"{base_url}/session/{session_id}/todo")
        assert resp.status_code == 200, (
            f"Expected 200 for GET /todo, got {resp.status_code}: {resp.text}"
        )
        todos = resp.json()
        assert isinstance(todos, list), f"Expected list response, got {type(todos)}: {todos}"


# ---------------------------------------------------------------------------
# C10.1 — GET /global/health
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subprocess_server", [_OPENCODE_PARAMS], indirect=True)
async def test_get_health(subprocess_server: SubprocessServer) -> None:
    """C10.1: GET /global/health, verify 200 with health status."""
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{base_url}/global/health")
        assert resp.status_code == 200, (
            f"Expected 200 for GET /global/health, got {resp.status_code}: {resp.text}"
        )
        health = resp.json()
        assert "healthy" in health, f"Expected 'healthy' field in health response: {health}"
        assert "version" in health, f"Expected 'version' field in health response: {health}"
        assert health["healthy"] is True, f"Expected healthy=True, got: {health['healthy']}"
        assert isinstance(health["version"], str), (
            f"Expected version to be a string, got: {type(health['version'])}"
        )


# ---------------------------------------------------------------------------
# C10.2 — POST /session/{session_id}/init
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="POST /init hangs in CI (fire-and-forget agent run blocks)",
    strict=False,
)
@pytest.mark.known_bug
@pytest.mark.parametrize("subprocess_server", [_OPENCODE_PARAMS], indirect=True)
async def test_post_init(subprocess_server: SubprocessServer) -> None:
    """C10.2: POST /session/{session_id}/init, verify 200."""
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=60.0) as client:
        session_id = await _create_session(client, base_url)

        # POST init — starts a fire-and-forget agent run to generate AGENTS.md.
        # Returns True immediately (the init task runs asynchronously).
        resp = await client.post(f"{base_url}/session/{session_id}/init", json={})
        assert resp.status_code == 200, (
            f"Expected 200 for POST /init, got {resp.status_code}: {resp.text}"
        )
        result = resp.json()
        # The endpoint returns True when the init task has been started.
        assert result is True, f"Expected True response from /init, got: {result}"
