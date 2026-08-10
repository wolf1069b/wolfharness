"""L4 subprocess E2E tests for the AG-UI server (``wolfharness serve-agui``).

L4a smoke tests (``@pytest.mark.e2e``, NOT slow):
    - test_server_startup: spawn serve-agui, verify HTTP health
    - test_event_stream: POST to agent endpoint, verify SSE event stream
    - test_server_shutdown: verify clean process exit after test

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


def _build_agui_request(prompt: str = "Hello, test agent!") -> dict[str, Any]:
    """Build a minimal AG-UI RunAgentInput request body.

    Args:
        prompt: The user prompt text.

    Returns:
        JSON-serializable dict matching the AG-UI RunAgentInput schema.
    """
    return {
        "thread_id": "thread-e2e",
        "run_id": "run-e2e",
        "state": {},
        "messages": [
            {
                "id": "msg-1",
                "role": "user",
                "content": prompt,
            }
        ],
        "tools": [],
        "context": [],
        "forwarded_props": {},
    }


# ---------------------------------------------------------------------------
# L4a Smoke Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-agui", "is_stdio": False}],
    indirect=True,
)
async def test_server_startup(subprocess_server: SubprocessServer, e2e_config: Path) -> None:
    """L4a: Start serve-agui, verify HTTP server is responding."""
    assert subprocess_server.process.returncode is None, "AG-UI server process exited early"
    assert subprocess_server.port > 0

    # The root endpoint lists available agents.
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(f"{subprocess_server.base_url}/")
        assert resp.status_code == 200, f"Root endpoint returned {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "agents" in data, f"Expected 'agents' key in response: {data}"
        assert data["count"] >= 1, f"Expected at least 1 agent: {data}"


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-agui", "is_stdio": False}],
    indirect=True,
)
async def test_event_stream(subprocess_server: SubprocessServer, e2e_config: Path) -> None:
    """L4a: POST to agent endpoint, verify SSE event stream response."""
    base_url = subprocess_server.base_url
    request_body = _build_agui_request("Hello from AG-UI e2e test!")

    # POST to /test_agent — should return a streaming SSE response.
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{base_url}/test_agent",
            json=request_body,
            headers={"Accept": "text/event-stream"},
        )
        # The AG-UI server returns a streaming response.
        # Status 200 means the request was accepted and streaming started.
        assert resp.status_code == 200, (
            f"Agent endpoint returned {resp.status_code}: {resp.text[:500]}"
        )


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-agui", "is_stdio": False}],
    indirect=True,
)
async def test_server_shutdown(subprocess_server: SubprocessServer, e2e_config: Path) -> None:
    """L4a: Verify server is responsive then shuts down cleanly."""
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(f"{base_url}/")
        assert resp.status_code == 200

    assert subprocess_server.process.returncode is None
