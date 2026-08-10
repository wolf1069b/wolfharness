"""L4 subprocess E2E tests for OpenCode prompt delivery modes.

Covers Phase C group C4 (2 tasks):
    - C4.1 test_steer_delivery — send prompt, then second with delivery:"steer"
    - C4.2 test_queue_delivery — send prompt, then second with delivery:"queue"

The OpenCode MessageRequest has a ``delivery`` field:
    - "steer" → maps to priority "asap" (injects into active turn)
    - anything else → maps to "when_idle" (queues for next turn)

All tests use ``model: test`` (pydantic-ai TestModel) so NO API key is needed.
L4a smoke tests: pytest -m "e2e and not slow" (~30s)
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
# Helpers
# ---------------------------------------------------------------------------


async def _create_session(base_url: str, client: httpx.AsyncClient) -> str:
    """Create a session and return its ID."""
    resp = await client.post(f"{base_url}/session", json={})
    assert resp.status_code in (200, 201), f"Failed to create session: {resp.status_code}"
    data = resp.json()
    return data.get("id") or data.get("sessionID")


# ---------------------------------------------------------------------------
# C4.1 — Steer delivery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_steer_delivery(
    subprocess_server: SubprocessServer,
    e2e_config: Path,
) -> None:
    """C4.1: Send prompt, immediately send second with delivery:"steer".

    The "steer" delivery mode maps to priority "asap", which injects the
    message into the active turn mid-execution via PydanticAI's
    PendingMessageDrainCapability.

    Verifies that the second message is accepted without error (200/201/202).
    With TestModel the first prompt completes near-instantly, so the steer
    message may be delivered as a follow-up rather than mid-turn. The key
    assertion is that the server accepts the delivery mode without error.
    """
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=30.0) as client:
        session_id = await _create_session(base_url, client)

        # Send first prompt (synchronous — waits for completion).
        first_payload: dict[str, Any] = {
            "parts": [{"type": "text", "text": "First prompt"}],
        }
        first_resp = await client.post(
            f"{base_url}/session/{session_id}/message",
            json=first_payload,
        )
        assert first_resp.status_code in (200, 201, 202), (
            f"First message failed: {first_resp.status_code}: {first_resp.text}"
        )

        # Immediately send second prompt with delivery="steer".
        steer_payload: dict[str, Any] = {
            "parts": [{"type": "text", "text": "Steer this conversation"}],
            "delivery": "steer",
        }
        steer_resp = await client.post(
            f"{base_url}/session/{session_id}/message",
            json=steer_payload,
        )
        # The server should accept the steer message.
        # It may return 200/201/202 (accepted) or an error if the session
        # state doesn't support steering at that moment.
        assert steer_resp.status_code in (200, 201, 202), (
            f"Steer delivery failed: {steer_resp.status_code}: {steer_resp.text}"
        )


# ---------------------------------------------------------------------------
# C4.2 — Queue delivery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_queue_delivery(
    subprocess_server: SubprocessServer,
    e2e_config: Path,
) -> None:
    """C4.2: Send prompt, immediately send second with delivery:"queue".

    The "queue" delivery mode maps to priority "when_idle", which queues the
    message for the next turn (processed after the current run completes).

    Verifies that the second message is accepted and waits for the next turn.
    With TestModel the first prompt completes quickly, so the queued message
    is processed in the subsequent turn. The key assertion is that the server
    accepts the queue delivery mode without error.
    """
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=30.0) as client:
        session_id = await _create_session(base_url, client)

        # Send first prompt (synchronous — waits for completion).
        first_payload: dict[str, Any] = {
            "parts": [{"type": "text", "text": "First prompt"}],
        }
        first_resp = await client.post(
            f"{base_url}/session/{session_id}/message",
            json=first_payload,
        )
        assert first_resp.status_code in (200, 201, 202), (
            f"First message failed: {first_resp.status_code}: {first_resp.text}"
        )

        # Immediately send second prompt with delivery="queue".
        queue_payload: dict[str, Any] = {
            "parts": [{"type": "text", "text": "Queue this for next turn"}],
            "delivery": "queue",
        }
        queue_resp = await client.post(
            f"{base_url}/session/{session_id}/message",
            json=queue_payload,
        )
        # The server should accept the queued message.
        assert queue_resp.status_code in (200, 201, 202), (
            f"Queue delivery failed: {queue_resp.status_code}: {queue_resp.text}"
        )
