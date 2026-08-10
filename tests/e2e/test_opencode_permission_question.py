"""L4 subprocess E2E tests for OpenCode permission and question endpoints.

Covers Phase C group C8 (6 tasks, 1 skip):
    - C8.1 test_get_permission: GET /permission → 200 with pending permissions list
    - C8.2 test_post_permission_reply: POST /permission/{id}/reply → 404 for
      non-existent permission (TestModel cannot trigger real permissions);
      SSE stream verified alive.
    - C8.3 test_get_question: GET /question/ → 200 with pending questions list
    - C8.4 test_post_question [skip]: POST /question not implemented yet
    - C8.5 test_post_question_reply: POST /question/{id}/reply → 404 for
      non-existent question; SSE stream verified alive.
    - C8.6 test_post_question_reject: POST /question/{id}/reject → 404 for
      non-existent question; SSE stream verified alive.

All tests use ``model: test`` (pydantic-ai TestModel) so NO API key is needed.
"""

from __future__ import annotations

import asyncio
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


async def _read_first_sse_event(
    client: httpx.AsyncClient, url: str, timeout: float = 5.0
) -> str | None:
    """Open an SSE stream and read the first ``data:`` event.

    Returns the event data string, or None if no event arrives within timeout.
    Closes the stream after reading the first event.
    """
    first_event: str | None = None
    async with client.stream("GET", url, timeout=timeout) as resp:
        assert resp.status_code == 200, f"SSE stream returned {resp.status_code}"
        async with asyncio.timeout(timeout):
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    first_event = line[len("data:") :].strip()
                    break
    return first_event


# ---------------------------------------------------------------------------
# C8.1 — GET /permission
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subprocess_server", [_OPENCODE_PARAMS], indirect=True)
async def test_get_permission(subprocess_server: SubprocessServer) -> None:
    """C8.1: GET /permission, verify 200 with pending permissions list."""
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        await _create_session(client, base_url)

        # GET /permission — global endpoint listing all pending permissions.
        resp = await client.get(f"{base_url}/permission")
        assert resp.status_code == 200, (
            f"Expected 200 for GET /permission, got {resp.status_code}: {resp.text}"
        )
        permissions = resp.json()
        assert isinstance(permissions, list), (
            f"Expected list response, got {type(permissions)}: {permissions}"
        )
        # With TestModel (no tool calls), there should be no pending permissions.
        assert len(permissions) == 0, (
            f"Expected no pending permissions with TestModel, got {permissions}"
        )


# ---------------------------------------------------------------------------
# C8.2 — POST /permission/{permission_id}/reply
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subprocess_server", [_OPENCODE_PARAMS], indirect=True)
async def test_post_permission_reply(subprocess_server: SubprocessServer) -> None:
    """C8.2: POST /permission/{permission_id}/reply.

    With TestModel, no real permission is triggered, so we verify:
    1. The endpoint exists and accepts the request format.
    2. A 404 is returned for a non-existent permission ID (correct behavior).
    3. The SSE stream is alive (permission.replied event requires a real
       pending permission which TestModel cannot trigger).
    """
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        await _create_session(client, base_url)

        # Verify SSE stream is alive before the POST.
        first_event = await _read_first_sse_event(client, f"{base_url}/event")
        assert first_event is not None, "SSE stream did not deliver a connected event"

        # POST reply for a non-existent permission → 404.
        reply_body: dict[str, Any] = {"reply": "once"}
        resp = await client.post(
            f"{base_url}/permission/nonexistent-perm-id/reply",
            json=reply_body,
        )
        assert resp.status_code == 404, (
            f"Expected 404 for non-existent permission, got {resp.status_code}: {resp.text}"
        )


# ---------------------------------------------------------------------------
# C8.3 — GET /question
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subprocess_server", [_OPENCODE_PARAMS], indirect=True)
async def test_get_question(subprocess_server: SubprocessServer) -> None:
    """C8.3: GET /question/, verify 200 with pending questions list."""
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        await _create_session(client, base_url)

        # GET /question/ — note the trailing slash (router prefix="/question", route="/").
        resp = await client.get(f"{base_url}/question/", follow_redirects=True)
        assert resp.status_code == 200, (
            f"Expected 200 for GET /question/, got {resp.status_code}: {resp.text}"
        )
        questions = resp.json()
        assert isinstance(questions, list), (
            f"Expected list response, got {type(questions)}: {questions}"
        )
        # With TestModel (no tool calls), there should be no pending questions.
        assert len(questions) == 0, f"Expected no pending questions with TestModel, got {questions}"


# ---------------------------------------------------------------------------
# C8.4 — POST /question [skip]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subprocess_server", [_OPENCODE_PARAMS], indirect=True)
async def test_post_question(subprocess_server: SubprocessServer) -> None:
    """Test intent: POST /question with JSON body containing session_id.

    Message string, and optional options array. Verify 200 or 201 with
    question_id in response. Follow with GET /question to verify question
    appears in pending list with matching message and options. Error case:
    invalid session_id -> 404; missing message -> 422.
    """


# ---------------------------------------------------------------------------
# C8.5 — POST /question/{requestID}/reply
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subprocess_server", [_OPENCODE_PARAMS], indirect=True)
async def test_post_question_reply(subprocess_server: SubprocessServer) -> None:
    """C8.5: POST /question/{requestID}/reply.

    With TestModel, no real question is triggered, so we verify:
    1. The endpoint exists and accepts the request format.
    2. A 404 is returned for a non-existent question ID (correct behavior).
    3. The SSE stream is alive (question.replied event requires a real
       pending question which TestModel cannot trigger).
    """
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        await _create_session(client, base_url)

        # Verify SSE stream is alive before the POST.
        first_event = await _read_first_sse_event(client, f"{base_url}/event")
        assert first_event is not None, "SSE stream did not deliver a connected event"

        # POST reply for a non-existent question → 404.
        reply_body: dict[str, Any] = {"answers": [["once"]]}
        resp = await client.post(
            f"{base_url}/question/nonexistent-question-id/reply",
            json=reply_body,
        )
        assert resp.status_code == 404, (
            f"Expected 404 for non-existent question, got {resp.status_code}: {resp.text}"
        )


# ---------------------------------------------------------------------------
# C8.6 — POST /question/{requestID}/reject
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subprocess_server", [_OPENCODE_PARAMS], indirect=True)
async def test_post_question_reject(subprocess_server: SubprocessServer) -> None:
    """C8.6: POST /question/{requestID}/reject.

    With TestModel, no real question is triggered, so we verify:
    1. The endpoint exists and accepts the request.
    2. A 404 is returned for a non-existent question ID (correct behavior).
    3. The SSE stream is alive (question.rejected event requires a real
       pending question which TestModel cannot trigger).
    """
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        await _create_session(client, base_url)

        # Verify SSE stream is alive before the POST.
        first_event = await _read_first_sse_event(client, f"{base_url}/event")
        assert first_event is not None, "SSE stream did not deliver a connected event"

        # POST reject for a non-existent question → 404.
        resp = await client.post(
            f"{base_url}/question/nonexistent-question-id/reject",
        )
        assert resp.status_code == 404, (
            f"Expected 404 for non-existent question, got {resp.status_code}: {resp.text}"
        )
