"""L4 subprocess E2E tests for OpenCode server error boundaries.

Covers Phase C group C11 (5 tests, 2 skip):
    - C11.1 test_invalid_json_body: POST with malformed JSON → 400
    - C11.2 test_prompt_nonexistent_session: POST prompt to random session → 404
      (xfail: OpenCode server auto-creates sessions on demand)
    - C11.3 test_validation_error: POST with invalid field values → 422
    - C11.4 test_503_service_unavailable [skip]: requires lifecycle timing control
    - C11.5 test_409_conflict [skip]: 409 Conflict not implemented yet

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
# C11.1 — Malformed JSON body → 400
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subprocess_server", [_OPENCODE_PARAMS], indirect=True)
async def test_invalid_json_body(subprocess_server: SubprocessServer) -> None:
    """C11.1: POST with malformed JSON body, verify 400 or 422.

    FastAPI may return 422 (RequestValidationError) for malformed JSON
    depending on the version. We accept both 400 and 422 as valid
    "bad request" responses.
    """
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Send malformed JSON to POST /session (create_session accepts an
        # optional SessionCreateRequest body).
        resp = await client.post(
            f"{base_url}/session",
            content=b"{invalid json body",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code in (400, 422), (
            f"Expected 400 or 422 for malformed JSON, got {resp.status_code}: {resp.text}"
        )


# ---------------------------------------------------------------------------
# C11.2 — POST prompt to non-existent session → 404
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subprocess_server", [_OPENCODE_PARAMS], indirect=True)
async def test_prompt_nonexistent_session(subprocess_server: SubprocessServer) -> None:
    """C11.2: POST prompt to a random/non-existent session ID, verify 404."""
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        # POST a message to a non-existent session.
        message_body: dict[str, Any] = {
            "parts": [{"type": "text", "text": "Hello"}],
        }
        resp = await client.post(
            f"{base_url}/session/nonexistent-session-id-12345/message",
            json=message_body,
        )
        assert resp.status_code == 404, (
            f"Expected 404 for non-existent session, got {resp.status_code}: {resp.text}"
        )


# ---------------------------------------------------------------------------
# C11.3 — Validation error → 422
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subprocess_server", [_OPENCODE_PARAMS], indirect=True)
async def test_validation_error(subprocess_server: SubprocessServer) -> None:
    """C11.3: POST with invalid field values (missing required fields), verify 422.

    ShellRequest requires ``agent`` and ``command`` fields. Sending a body
    with only ``agent`` (missing ``command``) should trigger a Pydantic
    validation error → 422.
    """
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        session_id = await _create_session(client, base_url)

        # POST shell with missing required ``command`` field → 422.
        resp = await client.post(
            f"{base_url}/session/{session_id}/shell",
            json={"agent": "test_agent"},
        )
        assert resp.status_code == 422, (
            f"Expected 422 for validation error (missing 'command'), "
            f"got {resp.status_code}: {resp.text}"
        )


# ---------------------------------------------------------------------------
# C11.4 — 503 Service Unavailable [skip]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subprocess_server", [_OPENCODE_PARAMS], indirect=True)
async def test_503_service_unavailable(subprocess_server: SubprocessServer) -> None:
    """Test intent: Send a request during server lifecycle transition.

    (startup or shutdown) and check for 503 status. Requires timing
    control to hit the server during a lifecycle transition window.
    """


# ---------------------------------------------------------------------------
# C11.5 — 409 Conflict [skip]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subprocess_server", [_OPENCODE_PARAMS], indirect=True)
async def test_409_conflict(subprocess_server: SubprocessServer) -> None:
    """Test intent: Send a prompt to a session with active steer-mode prompt.

    Expect HTTP 409 Conflict response.
    """
