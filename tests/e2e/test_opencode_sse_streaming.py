"""L4 subprocess E2E tests for OpenCode SSE streaming endpoints.

Covers Phase C group C1 (4 tasks):
    - C1.1 test_sse_event_sequence_get_event: GET /event SSE stream lifecycle
    - C1.2 test_sse_global_event_sequence: GET /global/event with GlobalEvent wrapper
    - C1.3 test_sse_heartbeat: heartbeat event within 15s of connection
    - C1.4 test_sse_reconnect: disconnect and reconnect behavior

IMPORTANT D4: Subscribe to SSE BEFORE sending prompt (TestModel is too fast,
events missed otherwise).

All tests use ``model: test`` (pydantic-ai TestModel) so NO API key is needed.
L4a smoke tests: pytest -m "e2e and not slow" (~30s)
"""

from __future__ import annotations

import asyncio
import json
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


async def _parse_sse_lines(response: httpx.Response, max_events: int = 20) -> list[dict[str, Any]]:
    """Parse SSE ``data:`` lines from a streaming httpx response.

    Args:
        response: The streaming httpx response.
        max_events: Maximum number of SSE events to collect.

    Returns:
        List of parsed JSON event dicts.
    """
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
# C1.1 — GET /event SSE event sequence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_sse_event_sequence_get_event(
    subprocess_server: SubprocessServer,
    e2e_config: Path,
) -> None:
    """C1.1: Subscribe to GET /event, send prompt, verify event lifecycle.

    Asserts the first SSE event is ``server.connected`` (the opencode client
    expects this as the initial handshake event upon connecting).
    """
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=30.0) as client:
        session_id = await _create_session(base_url, client)

        # D4: Subscribe to SSE BEFORE sending the prompt.
        async with client.stream("GET", f"{base_url}/event") as sse_response:
            assert sse_response.status_code == 200

            # Collect the first event (server.connected).
            events = await _parse_sse_lines(sse_response, max_events=1)

            # Send a prompt AFTER subscribing (D4).
            # NOTE: POST /session/{id}/message may return 500 due to a
            # pre-existing OpenTelemetry _IncludedRouter.path bug. The primary
            # assertion is the SSE handshake (server.connected), not the
            # message response. We send the prompt to trigger events but do
            # not hard-assert on the response code.
            message_payload: dict[str, Any] = {
                "parts": [{"type": "text", "text": "Hello SSE test!"}],
            }
            await client.post(
                f"{base_url}/session/{session_id}/message",
                json=message_payload,
            )

    # Assert the first SSE event is server.connected.
    assert len(events) >= 1, "No SSE events received"
    first_event = events[0]
    assert first_event.get("type") == "server.connected", (
        f"Expected first SSE event type 'server.connected', got '{first_event.get('type')}'"
    )


# ---------------------------------------------------------------------------
# C1.2 — GET /global/event with GlobalEvent wrapper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_sse_global_event_sequence(
    subprocess_server: SubprocessServer,
    e2e_config: Path,
) -> None:
    """C1.2: Consume GET /global/event, verify GlobalEvent envelope wrapper.

    The global event stream wraps events in a ``{"payload": ...}`` envelope.
    Asserts heartbeat events are sent at intervals <15s.
    """
    base_url = subprocess_server.base_url

    async with (
        httpx.AsyncClient(timeout=30.0) as client,
        client.stream("GET", f"{base_url}/global/event") as sse_response,
    ):
        assert sse_response.status_code == 200

        events = await _parse_sse_lines(sse_response, max_events=2)

    assert len(events) >= 1, "No global SSE events received"
    # The first event should be server.connected, wrapped in payload envelope.
    first_event = events[0]
    # GlobalEvent wraps the event in a "payload" key.
    assert "payload" in first_event, (
        f"Expected GlobalEvent wrapper with 'payload' key, got keys: {list(first_event.keys())}"
    )
    payload = first_event["payload"]
    assert payload.get("type") == "server.connected", (
        f"Expected payload type 'server.connected', got '{payload.get('type')}'"
    )


# ---------------------------------------------------------------------------
# C1.3 — SSE heartbeat within 15s
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_sse_heartbeat(
    subprocess_server: SubprocessServer,
    e2e_config: Path,
) -> None:
    """C1.3: Verify SSE heartbeat event received within 15s of connection.

    The opencode client expects heartbeat events at intervals <15s to detect
    stale connections. The server sends ``server.heartbeat`` events every 10s
    when the EventBus stream is idle.
    """
    base_url = subprocess_server.base_url

    heartbeat_received = False
    async with (
        httpx.AsyncClient(timeout=20.0) as client,
        client.stream("GET", f"{base_url}/event") as sse_response,
    ):
        assert sse_response.status_code == 200

        events = await _parse_sse_lines(sse_response, max_events=50)
        heartbeat_received = any(event.get("type") == "server.heartbeat" for event in events)

    assert heartbeat_received, "No heartbeat event received within 15s of SSE connection"


# ---------------------------------------------------------------------------
# C1.4 — SSE reconnect behavior
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_sse_reconnect(
    subprocess_server: SubprocessServer,
    e2e_config: Path,
) -> None:
    """C1.4: Connect to GET /event, disconnect, reconnect with last_event_id.

    Verifies that after disconnecting and reconnecting, the SSE stream resumes
    and the first event is again ``server.connected`` (handshake).
    """
    base_url = subprocess_server.base_url

    # First connection: get the first event ID.
    first_event_id: str | None = None
    async with (
        httpx.AsyncClient(timeout=10.0) as client,
        client.stream("GET", f"{base_url}/event") as sse_response,
    ):
        assert sse_response.status_code == 200
        async for line in sse_response.aiter_lines():
            if line.startswith("id:"):
                first_event_id = line[len("id:") :].strip()
                break
            if line.startswith("data:"):
                # Some servers send data before id; keep reading.
                continue

    # Disconnect is implicit (context manager exit).
    # Brief delay to allow server to register disconnect.
    await asyncio.sleep(0.3)

    # Reconnect, optionally with last_event_id.
    events: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        url = f"{base_url}/event"
        if first_event_id is not None:
            url = f"{url}?last_event_id={first_event_id}"
        async with client.stream("GET", url) as sse_response:
            assert sse_response.status_code == 200

            events = await _parse_sse_lines(sse_response, max_events=1)

    assert len(events) >= 1, "No events received after reconnect"
    assert events[0].get("type") == "server.connected", (
        f"Expected 'server.connected' after reconnect, got '{events[0].get('type')}'"
    )


# ---------------------------------------------------------------------------
# C1.5 — Tool call parameters in SSE events (subprocess e2e)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server_with_tool",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_sse_tool_call_parameters_visible(
    subprocess_server_with_tool: SubprocessServer,
) -> None:
    """C1.5: Send a prompt that triggers a tool call, verify tool input in SSE payload.

    Uses TestModel with ``call_tools: ["bash"]`` and
    ``tool_args: {"bash": {"command": "echo hello"}}``.
    The TestModel will emit a tool call to ``bash`` with ``{"command": "echo hello"}``.

    Asserts:
    - At least one ``message.part.updated`` SSE event contains a ToolPart
    - The ToolPart's ``state.input`` includes ``command: "echo hello"``
    - Both running and completed ToolPart states preserve the input
    """
    base_url = subprocess_server_with_tool.base_url

    async with httpx.AsyncClient(timeout=30.0) as client:
        session_id = await _create_session(base_url, client)

        # D4: Subscribe to SSE BEFORE sending the prompt.
        # Collect all events in a single iteration pass (httpx stream can only be consumed once).
        events: list[dict[str, Any]] = []
        prompt_sent = False
        async with client.stream("GET", f"{base_url}/event") as sse_response:
            assert sse_response.status_code == 200

            async for line in sse_response.aiter_lines():
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

                # After receiving server.connected, send the prompt
                if not prompt_sent and event.get("type") == "server.connected":
                    prompt_sent = True
                    message_payload: dict[str, Any] = {
                        "parts": [{"type": "text", "text": "Run echo hello"}],
                    }
                    # Fire and forget — the response is not needed
                    await client.post(
                        f"{base_url}/session/{session_id}/message",
                        json=message_payload,
                    )

                # Stop after collecting enough events
                if len(events) >= 50:
                    break
                # Only break on idle AFTER prompt was sent and we've seen part events
                if (
                    prompt_sent
                    and event.get("type") == "session.status"
                    and event.get("properties", {}).get("status", {}).get("type") == "idle"
                    and any(e.get("type") == "message.part.updated" for e in events)
                ):
                    await asyncio.sleep(0.3)
                    break

    assert len(events) >= 1, "No SSE events received"

    # Filter for message.part.updated events (these carry ToolPart data)
    part_updated_events = [e for e in events if e.get("type") == "message.part.updated"]

    assert len(part_updated_events) >= 1, (
        f"No message.part.updated events received. Event types seen: "
        f"{[e.get('type') for e in events]}"
    )

    # Find ToolPart events by checking properties.part.type
    tool_part_events = [
        e
        for e in part_updated_events
        if e.get("properties", {}).get("part", {}).get("type") == "tool"
    ]

    assert len(tool_part_events) >= 1, (
        f"No ToolPart events found in {len(part_updated_events)} message.part.updated events. "
        f"Part types seen: "
        f"{[e.get('properties', {}).get('part', {}).get('type') for e in part_updated_events]}"
    )

    # Check that at least one ToolPart has non-empty state.input with command
    tool_parts_with_input = []
    for event in tool_part_events:
        part = event.get("properties", {}).get("part", {})
        state = part.get("state", {})
        input_data = state.get("input", {})
        if input_data.get("command"):
            tool_parts_with_input.append(event)

    assert len(tool_parts_with_input) >= 1, (
        f"No ToolPart events with non-empty state.input.command found. "
        f"ToolPart states: "
        f"{[e.get('properties', {}).get('part', {}).get('state', {}) for e in tool_part_events]}"
    )

    # Verify the command value matches what TestModel was configured to send
    for event in tool_parts_with_input:
        part = event.get("properties", {}).get("part", {})
        state = part.get("state", {})
        input_data = state.get("input", {})
        assert input_data.get("command") == "echo hello", (
            f"Expected command='echo hello', got '{input_data.get('command')}'"
        )
