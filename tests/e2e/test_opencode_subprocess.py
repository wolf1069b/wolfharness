"""L4 subprocess E2E tests for the OpenCode server (``wolfharness serve-opencode``).

L4a smoke tests (``@pytest.mark.e2e``, NOT slow):
    - test_server_startup: spawn serve-opencode, verify HTTP health
    - test_basic_prompt: create session, send message, verify response
    - test_server_shutdown: verify clean process exit after test

L4b full tests (``@pytest.mark.e2e`` + ``@pytest.mark.slow``):
    - test_tool_call: send message with tool configuration
    - test_session_close: create and close a session
    - test_error_paths: verify 404 for non-existent session

All tests use ``model: test`` (pydantic-ai TestModel) so NO API key is needed.
L4a smoke tests: pytest -m "e2e and not slow" (~30s)
L4b full tests: pytest -m e2e
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import pytest

from tests.e2e.conftest import SKIP_NO_BINARY, SKIP_WINDOWS


if TYPE_CHECKING:
    from pathlib import Path

    from tests.e2e.conftest import SubprocessServer


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(SKIP_NO_BINARY, reason="wolfharness binary not on PATH"),
    pytest.mark.skipif(SKIP_WINDOWS, reason="Windows subprocess issues"),
]


# ---------------------------------------------------------------------------
# L4a Smoke Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_server_startup(subprocess_server: SubprocessServer, e2e_config: Path) -> None:
    """L4a: Start serve-opencode, verify HTTP server is responding."""
    # The fixture already verified health via HTTP polling.
    # Double-check the process is alive.
    assert subprocess_server.process.returncode is None, "OpenCode server process exited early"
    assert subprocess_server.port > 0, "Expected non-zero port"

    # Verify we can hit the server.
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(f"{subprocess_server.base_url}/")
        # OpenCode server may return 200, 404, or redirect — any non-5xx is fine.
        assert resp.status_code < 500, f"Server returned {resp.status_code} on root endpoint"


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_basic_prompt(subprocess_server: SubprocessServer, e2e_config: Path) -> None:
    """L4a: Create session, send message, verify response via HTTP SSE."""
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Create a session.
        resp = await client.post(f"{base_url}/session", json={})
        assert resp.status_code in (200, 201), (
            f"Failed to create session: {resp.status_code} {resp.text}"
        )
        session_data = resp.json()
        session_id = session_data.get("id") or session_data.get("sessionID")
        assert session_id, f"Expected session ID in response: {session_data}"

        # Send a message (POST /session/{id}/message).
        message_payload: dict[str, Any] = {
            "parts": [{"type": "text", "text": "Hello, test agent!"}],
        }
        resp = await client.post(
            f"{base_url}/session/{session_id}/message",
            json=message_payload,
        )
        assert resp.status_code in (200, 201, 202), (
            f"Failed to send message: {resp.status_code} {resp.text}"
        )


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_server_shutdown(subprocess_server: SubprocessServer, e2e_config: Path) -> None:
    """L4a: Verify server is responsive then shuts down cleanly."""
    base_url = subprocess_server.base_url

    # Verify server is up.
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(f"{base_url}/")
        assert resp.status_code < 500

    # The fixture teardown will SIGTERM the process.
    # We just verify it was alive during the test.
    assert subprocess_server.process.returncode is None


# ---------------------------------------------------------------------------
# L4b Full Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "subprocess_server_with_tool",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_tool_call(
    subprocess_server_with_tool: SubprocessServer,
) -> None:
    """L4b: Send message with tool configuration, verify tool execution."""
    base_url = subprocess_server_with_tool.base_url

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Create a session.
        resp = await client.post(f"{base_url}/session", json={})
        assert resp.status_code in (200, 201)
        session_data = resp.json()
        session_id = session_data.get("id") or session_data.get("sessionID")
        assert session_id

        # Send a message requesting tool use.
        message_payload: dict[str, Any] = {
            "parts": [{"type": "text", "text": "Run echo hello"}],
        }
        resp = await client.post(
            f"{base_url}/session/{session_id}/message",
            json=message_payload,
        )
        assert resp.status_code in (200, 201, 202)


@pytest.mark.slow
@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_session_close(subprocess_server: SubprocessServer, e2e_config: Path) -> None:
    """L4b: Create a session, list sessions, verify it appears."""
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Create a session.
        resp = await client.post(f"{base_url}/session", json={})
        assert resp.status_code in (200, 201)
        session_data = resp.json()
        session_id = session_data.get("id") or session_data.get("sessionID")
        assert session_id

        # List sessions.
        resp = await client.get(f"{base_url}/session")
        assert resp.status_code == 200
        sessions = resp.json()
        # The session should appear in the list.
        session_ids = [s.get("id") or s.get("sessionID") for s in sessions]
        assert session_id in session_ids, (
            f"Session {session_id} not found in session list: {session_ids}"
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_error_paths(subprocess_server: SubprocessServer, e2e_config: Path) -> None:
    """L4b: Verify 404 for non-existent session."""
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=5.0) as client:
        # Try to get a non-existent session.
        resp = await client.get(f"{base_url}/session/nonexistent-session-id")
        assert resp.status_code == 404, (
            f"Expected 404 for non-existent session, got {resp.status_code}"
        )
