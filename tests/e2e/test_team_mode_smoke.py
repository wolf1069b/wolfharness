"""L4a smoke E2E test for dynamic team mode via ``wolfharness serve-opencode``.

Spawns a real ``wolfharness serve-opencode`` subprocess with a YAML config
that enables ``team_mode`` and uses ``model: test`` (pydantic-ai
TestModel — no API key needed). Verifies the server starts, accepts
connections, and emits the SSE handshake event.

This is an **L4a smoke test** (``@pytest.mark.e2e``, NOT
``@pytest.mark.slow``). It runs in ~30s and catches "server won't start
with team_mode" regressions.

**Known limitation**: POST ``/session/{id}/message`` returns 500 due to
a pre-existing OpenTelemetry ``_IncludedRouter.path`` bug (issues #185,
#190). The primary assertion is the SSE ``server.connected`` handshake,
which proves the server starts correctly with ``team_mode`` enabled.
Prompt delivery is tested via ``@pytest.mark.xfail`` until the OTel bug
is fixed.

See ``tests/AGENTS.md`` § "L4 Sub-layers" and
``openspec/changes/layered-testing-infrastructure/design.md`` D17.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import yaml

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
# Team-mode config (model: test — no API key needed)
# ---------------------------------------------------------------------------

_TEAM_MODE_CONFIG = {
    "agents": {
        "team_lead": {
            "type": "native",
            "model": "test",
            "system_prompt": "You are a team lead. Use team tools to coordinate.",
        },
        "team_member": {
            "type": "native",
            "model": "test",
            "system_prompt": "You are a team member.",
        },
    },
    "team_mode": {
        "enabled": True,
        "lead_eligible": ["team_lead"],
        "member_eligible": ["team_lead", "team_member"],
    },
}


@pytest.fixture
def team_mode_e2e_config(tmp_path: Path) -> Path:
    """Write team-mode YAML config to tmp_path for subprocess server."""
    config_path = tmp_path / "team_mode_e2e.yml"
    config_path.write_text(yaml.dump(_TEAM_MODE_CONFIG, default_flow_style=False))
    return config_path


# ---------------------------------------------------------------------------
# SSE helpers (matching test_opencode_sse_streaming.py pattern)
# ---------------------------------------------------------------------------


async def _parse_sse_lines(response: httpx.Response, max_events: int = 20) -> list[dict[str, Any]]:
    """Parse SSE ``data:`` lines from a streaming httpx response."""
    events: list[dict[str, Any]] = []
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        data_str = line[len("data:") :].strip()
        if not data_str:
            continue
        try:
            event = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        events.append(event)
        if len(events) >= max_events:
            break
    return events


async def _create_session(base_url: str, client: httpx.AsyncClient) -> str:
    """Create a session and return its ID."""
    resp = await client.post(f"{base_url}/session", json={})
    assert resp.status_code in (200, 201), f"Failed to create session: {resp.status_code}"
    data = resp.json()
    return data.get("id") or data.get("sessionID")


# ---------------------------------------------------------------------------
# L4a smoke test: server starts with team_mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [
        {
            "serve_command": "serve-opencode",
            "is_stdio": False,
            "health_path": "/session",
            "health_timeout": 15.0,
        }
    ],
    indirect=True,
)
async def test_team_mode_server_starts(
    subprocess_server: SubprocessServer,
    team_mode_e2e_config: Path,
) -> None:
    """L4a smoke: serve-opencode starts successfully with team_mode enabled.

    Given: a YAML config with ``team_mode`` enabled and ``model: test``.

    When: ``wolfharness serve-opencode`` is spawned as a subprocess.

    Then:
    - The server starts and responds to HTTP health checks.
    - A session can be created (proves the team_mode config loaded).
    - The SSE ``/event`` stream emits a ``server.connected`` handshake.
    """
    base_url = subprocess_server.base_url

    # --- 1. Verify server is alive ----------------------------------------
    assert subprocess_server.process.returncode is None, "serve-opencode process exited early"

    async with httpx.AsyncClient(timeout=30.0) as client:
        # --- 2. Create a session ------------------------------------------
        session_id = await _create_session(base_url, client)
        assert session_id, "Expected session ID"

        # --- 3. Subscribe to SSE BEFORE any prompt (D4: TestModel is fast) -
        # The server.connected event is emitted when the SSE connection opens.
        async with client.stream("GET", f"{base_url}/event") as sse_response:
            assert sse_response.status_code == 200, (
                f"Expected 200 from /event, got {sse_response.status_code}"
            )

            # Collect the first SSE event (server.connected handshake).
            events = await _parse_sse_lines(sse_response, max_events=1)

    # --- 4. Assert SSE handshake was received -----------------------------
    assert len(events) >= 1, (
        "No SSE events received from /event stream — "
        "server may not be emitting the server.connected handshake"
    )
    first_event = events[0]
    assert first_event.get("type") == "server.connected", (
        f"Expected first SSE event type 'server.connected', got '{first_event.get('type')}'"
    )

    # --- 5. Verify process is still alive (clean state) -------------------
    assert subprocess_server.process.returncode is None, (
        "serve-opencode process died during the test"
    )


# ---------------------------------------------------------------------------
# L4a smoke test: session creation with team_mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [
        {
            "serve_command": "serve-opencode",
            "is_stdio": False,
            "health_path": "/session",
            "health_timeout": 15.0,
        }
    ],
    indirect=True,
)
async def test_team_mode_session_creation(
    subprocess_server: SubprocessServer,
    team_mode_e2e_config: Path,
) -> None:
    """L4a smoke: create a session on a team_mode-enabled server.

    Given: serve-opencode running with team_mode + TestModel.

    When: POST /session is called.

    Then: session is created (200/201) with a valid session ID.
    """
    base_url = subprocess_server.base_url
    assert subprocess_server.process.returncode is None

    async with httpx.AsyncClient(timeout=10.0) as client:
        session_id = await _create_session(base_url, client)
        assert session_id, f"Expected session ID, got: {session_id}"


# ---------------------------------------------------------------------------
# Prompt delivery (xfail — known OTel bug #185/#190)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [
        {
            "serve_command": "serve-opencode",
            "is_stdio": False,
            "health_path": "/session",
            "health_timeout": 15.0,
        }
    ],
    indirect=True,
)
@pytest.mark.xfail(
    reason=(
        "POST /session/{id}/message returns 500 due to OTel _IncludedRouter.path bug (#185, #190)"
    ),
    strict=False,
    raises=AssertionError,
)
@pytest.mark.known_bug
async def test_team_mode_prompt_delivery(
    subprocess_server: SubprocessServer,
    team_mode_e2e_config: Path,
) -> None:
    """L4a: send a prompt to a team_mode-enabled server.

    XFAIL: POST /session/{id}/message returns 500 due to a pre-existing
    OpenTelemetry FastAPI instrumentation bug (_IncludedRouter has no
    attribute 'path'). This is NOT a team_mode bug — the same issue
    affects all serve-opencode E2E tests.

    Once the OTel bug is fixed, this test will pass automatically
    (strict=False) and can be moved to a regular assertion.
    """
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=30.0) as client:
        session_id = await _create_session(base_url, client)

        message_payload: dict[str, Any] = {
            "parts": [{"type": "text", "text": "Hello, team lead!"}],
        }
        resp = await client.post(
            f"{base_url}/session/{session_id}/message",
            json=message_payload,
        )
        assert resp.status_code in (200, 201, 202), (
            f"Expected 200/201/202 from message, got {resp.status_code}: {resp.text}"
        )
