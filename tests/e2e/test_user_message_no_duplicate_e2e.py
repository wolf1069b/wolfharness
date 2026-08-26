"""L4a E2E test: user message renders exactly once via prompt_async.

Reproduces the double-rendering bug where a single ``POST /prompt_async``
causes the user message to appear twice in the TUI.

The test verifies via both:
    - SSE event stream (count ``message.updated`` events for user role)
    - Messages REST endpoint (``GET /session/{id}/message``)

Strategy:
    1. Start a real ``wolfharness serve-opencode`` subprocess with a TestModel.
    2. Open the SSE event stream (``GET /event``).
    3. Send ONE prompt via ``POST /prompt_async``.
    4. Collect SSE events until the session goes idle.
    5. Assert exactly 1 unique user ``message.updated`` event by message ID.
    6. Assert exactly 1 user message in the messages REST endpoint.

L4a smoke: ``pytest -m "e2e and not slow"`` (~15s).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from tests.e2e.conftest import SKIP_NO_BINARY, SKIP_WINDOWS, _spawn_server


if TYPE_CHECKING:
    from pathlib import Path

    from tests.e2e.conftest import SubprocessServer


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(SKIP_NO_BINARY, reason="wolfharness binary not on PATH"),
    pytest.mark.skipif(SKIP_WINDOWS, reason="Windows subprocess issues"),
]


# ---------------------------------------------------------------------------
# Config — simple TestModel, no tools
# ---------------------------------------------------------------------------

SIMPLE_CONFIG_YAML = """
agents:
  test_agent:
    type: native
    model:
      type: test
      custom_output_text: "Hello"
    system_prompt: "You are a test assistant."
storage:
  providers:
    - type: memory
"""


@pytest.fixture
def simple_config(tmp_path: Path) -> Path:
    """YAML config with a simple TestModel agent."""
    config_path = tmp_path / "simple_config.yml"
    config_path.write_text(SIMPLE_CONFIG_YAML.strip() + "\n")
    return config_path


@pytest.fixture
async def subprocess_server_simple(
    process_registry: Any,
    simple_config: Path,
    allow_model_requests: Any,
) -> Any:
    """Spawn an opencode server with the simple config (non-cached)."""
    async for server in _spawn_server(
        "serve-opencode",
        simple_config,
        process_registry=process_registry,
        is_stdio=False,
        health_path="/session",
        health_timeout=15.0,
    ):
        yield server


# ---------------------------------------------------------------------------
# Helpers (duplicated from test_turn_split_e2e.py for isolation)
# ---------------------------------------------------------------------------


async def _create_session(base_url: str, client: httpx.AsyncClient) -> str:
    """Create a session and return its ID."""
    resp = await client.post(f"{base_url}/session", json={})
    assert resp.status_code in (200, 201), f"Failed to create session: {resp.status_code}"
    data = resp.json()
    return data.get("id") or data.get("sessionID")


def _parse_sse_line(line: str) -> dict[str, Any] | None:
    """Parse a single SSE ``data:`` line into a JSON dict, or None."""
    if not line.startswith("data:"):
        return None
    data_str = line[len("data:") :].strip()
    if not data_str:
        return None
    try:
        return json.loads(data_str)
    except json.JSONDecodeError:
        return None


def _get_message_updated_info(
    event: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract ``properties.info`` from a ``message.updated`` event, or None."""
    if event.get("type") != "message.updated":
        return None
    return event.get("properties", {}).get("info")


def _is_session_idle(event: dict[str, Any]) -> bool:
    """Check if a SSE event is ``session.status`` with ``type=idle``."""
    if event.get("type") != "session.status":
        return False
    status_obj = event.get("properties", {}).get("status", {})
    return status_obj.get("type") == "idle"


def _is_session_busy(event: dict[str, Any]) -> bool:
    """Check if a SSE event is ``session.status`` with ``type=busy``."""
    if event.get("type") != "session.status":
        return False
    status_obj = event.get("properties", {}).get("status", {})
    return status_obj.get("type") == "busy"


# ---------------------------------------------------------------------------
# E2E test
# ---------------------------------------------------------------------------


async def test_prompt_async_user_message_renders_once(  # noqa: PLR0915
    subprocess_server_simple: SubprocessServer,
) -> None:
    """L4a: Verify a single prompt_async produces exactly 1 user message.

    This test reproduces the double-rendering bug where a single
    ``POST /prompt_async`` causes the user message to appear twice.
    """
    base_url = subprocess_server_simple.base_url
    prompt_text = "DUPLICATE_CHECK_MARKER"

    async with (
        httpx.AsyncClient(timeout=60.0) as sse_client,
        httpx.AsyncClient(timeout=60.0) as http_client,
    ):
        session_id = await _create_session(base_url, http_client)

        # --- Concurrent SSE collection + prompt sending ---
        sse_events: list[dict[str, Any]] = []
        saw_busy = asyncio.Event()

        async def collect_sse_events() -> None:
            """Collect SSE events until session goes idle (after run) or timeout."""
            deadline = time.monotonic() + 30.0
            async with sse_client.stream("GET", f"{base_url}/event") as sse_response:
                assert sse_response.status_code == 200
                async for line in sse_response.aiter_lines():
                    event = _parse_sse_line(line)
                    if event is not None:
                        sse_events.append(event)
                        if _is_session_busy(event):
                            saw_busy.set()
                        if _is_session_idle(event) and saw_busy.is_set():
                            break
                    if time.monotonic() > deadline:
                        break

        async def send_prompt() -> None:
            """Send a single prompt via prompt_async."""
            await asyncio.sleep(0.5)  # Wait for SSE to connect
            resp = await http_client.post(
                f"{base_url}/session/{session_id}/prompt_async",
                json={"parts": [{"type": "text", "text": prompt_text}]},
            )
            assert resp.status_code == 204, f"prompt_async failed: {resp.status_code}: {resp.text}"

        sse_task = asyncio.create_task(collect_sse_events(), name="sse_collector")
        prompt_task = asyncio.create_task(send_prompt(), name="prompt_sender")

        try:
            await asyncio.wait_for(asyncio.gather(sse_task, prompt_task), timeout=35.0)
        except TimeoutError:
            pytest.fail(
                f"Timed out waiting for SSE collection. "
                f"Collected {len(sse_events)} events. "
                f"Event types: {[e.get('type') for e in sse_events]}"
            )

        # --- Verify via messages REST endpoint ---
        # Poll until we have messages or timeout
        deadline = time.monotonic() + 10.0
        messages: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            resp = await http_client.get(f"{base_url}/session/{session_id}/message")
            if resp.status_code == 200:
                messages = resp.json()
                if len(messages) >= 2:  # At least 1 user + 1 assistant
                    break
            await asyncio.sleep(0.5)

    # --- Assertion 1: Exactly 1 user message in REST endpoint ---
    user_messages_rest = [m for m in messages if m.get("info", {}).get("role") == "user"]
    user_msg_ids_rest = [m["info"]["id"] for m in user_messages_rest]

    assert len(user_msg_ids_rest) == 1, (
        f"Expected exactly 1 user message in REST endpoint, "
        f"got {len(user_msg_ids_rest)}: {user_msg_ids_rest}. "
        f"All message roles: {[m.get('info', {}).get('role') for m in messages]}"
    )

    # --- Assertion 2: Exactly 1 unique user message.updated in SSE ---
    message_updated_infos = [
        info for event in sse_events if (info := _get_message_updated_info(event)) is not None
    ]
    sse_user_msgs = [info for info in message_updated_infos if info.get("role") == "user"]

    # Deduplicate by message ID (replay buffer may send duplicates)
    seen_user_ids: list[str] = []
    for info in sse_user_msgs:
        msg_id = info["id"]
        if msg_id not in seen_user_ids:
            seen_user_ids.append(msg_id)

    assert len(seen_user_ids) == 1, (
        f"Expected exactly 1 unique user message.updated event in SSE, "
        f"got {len(seen_user_ids)}: {seen_user_ids}. "
        f"All user message infos: {sse_user_msgs}"
    )

    # --- Assertion 3: SSE and REST agree on the message ID ---
    assert seen_user_ids[0] == user_msg_ids_rest[0], (
        f"SSE user message ID ({seen_user_ids[0]}) != REST user message ID ({user_msg_ids_rest[0]})"
    )

    # --- Assertion 4: The user message content appears in SSE (message.part.updated) ---
    # REST API no longer includes user message parts (stripped to prevent TUI duplication).
    # SSE message.part.updated is the sole source of user message content.
    sse_part_updated_events = [
        event for event in sse_events if event.get("type") == "message.part.updated"
    ]
    sse_user_text_parts = [
        event.get("properties", {}).get("part", {}).get("text", "")
        for event in sse_part_updated_events
        if event.get("properties", {}).get("part", {}).get("type") == "text"
    ]
    user_part_found = any(prompt_text in text for text in sse_user_text_parts)
    assert user_part_found, (
        f"Prompt text '{prompt_text}' not found in SSE message.part.updated events. "
        f"Found {len(sse_part_updated_events)} part.updated events. "
        f"Text parts from SSE: {sse_user_text_parts}"
    )


async def test_attach_existing_session_first_prompt_renders_once(
    subprocess_server_simple: SubprocessServer,
) -> None:
    """L4a: Attach to a pre-existing session, first prompt renders once.

    Reproduces issue #380 exactly: ``serve-opencode`` + ``opencode attach``
    against a session that already has history. The session consumer is
    running and the replay buffer is populated; the first prompt on the
    attached stream must produce exactly ONE ``message.updated`` per
    ``message_id`` (no EventBus loopback, no replay re-conversion).
    """
    base_url = subprocess_server_simple.base_url
    marker_a = "PRE_EXISTING_TURN"
    marker_b = "ATTACH_FIRST_PROMPT"

    async with (
        httpx.AsyncClient(timeout=60.0) as sse_client,
        httpx.AsyncClient(timeout=60.0) as http_client,
    ):
        session_id = await _create_session(base_url, http_client)

        # Turn 1: pre-existing history on the session.
        resp = await http_client.post(
            f"{base_url}/session/{session_id}/prompt_async",
            json={"parts": [{"type": "text", "text": marker_a}]},
        )
        assert resp.status_code == 204, f"first prompt_async failed: {resp.status_code}"
        # Wait for the session to go idle (turn complete).
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            resp = await http_client.get(f"{base_url}/session/{session_id}")
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "idle":
                    break
            await asyncio.sleep(0.5)

        # Turn 2: attach a fresh SSE stream (the "attach" connection) and send
        # the first prompt on it.
        sse_events: list[dict[str, Any]] = []
        saw_busy = asyncio.Event()

        async def collect_sse_events() -> None:
            deadline = time.monotonic() + 30.0
            async with sse_client.stream("GET", f"{base_url}/event") as sse_response:
                assert sse_response.status_code == 200
                async for line in sse_response.aiter_lines():
                    event = _parse_sse_line(line)
                    if event is not None:
                        sse_events.append(event)
                        if _is_session_busy(event):
                            saw_busy.set()
                        if _is_session_idle(event) and saw_busy.is_set():
                            break
                    if time.monotonic() > deadline:
                        break

        async def send_prompt() -> None:
            await asyncio.sleep(0.5)  # Wait for SSE to connect
            resp = await http_client.post(
                f"{base_url}/session/{session_id}/prompt_async",
                json={"parts": [{"type": "text", "text": marker_b}]},
            )
            assert resp.status_code == 204, (
                f"attach prompt_async failed: {resp.status_code}: {resp.text}"
            )

        sse_task = asyncio.create_task(collect_sse_events(), name="attach_sse_collector")
        prompt_task = asyncio.create_task(send_prompt(), name="attach_prompt_sender")

        try:
            await asyncio.wait_for(asyncio.gather(sse_task, prompt_task), timeout=35.0)
        except TimeoutError:
            pytest.fail(
                f"Timed out waiting for attach SSE collection. "
                f"Collected {len(sse_events)} events. "
                f"Event types: {[e.get('type') for e in sse_events]}"
            )

    # --- Assertion: exactly 1 message.updated per user message_id on attach ---
    message_updated_infos = [
        info for event in sse_events if (info := _get_message_updated_info(event)) is not None
    ]
    sse_user_msgs = [info for info in message_updated_infos if info.get("role") == "user"]

    from collections import Counter

    user_id_counts = Counter(info["id"] for info in sse_user_msgs)
    assert user_id_counts, "No user message.updated events collected on attach stream"

    duplicate_ids = {mid for mid, count in user_id_counts.items() if count > 1}
    assert not duplicate_ids, (
        f"Duplicate message.updated deliveries on attach for message_ids: {duplicate_ids}. "
        f"All user message infos: {sse_user_msgs}"
    )

    # The attach stream connects after turn 1 (replay=False on first connect),
    # so it should see exactly ONE user message (turn 2's first prompt) and
    # that message must be delivered exactly once.
    assert len(user_id_counts) == 1, (
        f"Attach stream should see exactly 1 user message, got {len(user_id_counts)}. "
        f"All user message infos: {sse_user_msgs}"
    )
