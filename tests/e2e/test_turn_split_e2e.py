"""L4a E2E test for logical turn split via steer message.

Verifies that when a steer ``UserMessageInsertedEvent`` arrives during an
active turn, the next ``PartStartEvent`` triggers a logical turn split:
the current assistant message (A1) is finalized and a new assistant
message (A2) is created with a fresh message ID, so the steer user
message sorts between A1 and A2 in the TUI's lexicographic message ID
ordering.

The test verifies via both:
    - SSE event stream (``message.updated`` events with ordering)
    - Messages REST endpoint (``GET /session/{id}/message``)

Strategy:
    1. Start a real ``wolfharness serve-opencode`` subprocess with a TestModel
       configured to call the ``bash`` tool (``sleep 2``) on the first
       request.  This gives a 2-second window during tool execution for
       the steer to arrive mid-turn.
    2. Open the SSE event stream (``GET /event``) on a separate HTTP client.
    3. Send the first prompt via ``POST /prompt_async`` (non-blocking).
    4. Wait 1 second for the model to respond and the tool to start.
    5. Send the steer via ``POST /prompt_async`` with ``delivery="steer"``.
    6. Collect SSE events until the session goes idle (after the run).
    7. Assert 2 assistant ``message.updated`` events with different IDs.
    8. Assert 1 steer user ``message.updated`` event between them.
    9. Assert message ID ordering: ``A1 < steer < A2``.

L4a smoke: ``pytest -m "e2e and not slow"`` (~30s).
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
# Config — TestModel that calls bash(sleep 2) on first request
# ---------------------------------------------------------------------------

TURN_SPLIT_CONFIG_YAML = """
agents:
  test_agent:
    type: native
    model:
      type: test
      custom_output_text: "Test response"
      call_tools: ["bash"]
      tool_args:
        bash:
          command: "sleep 2"
    system_prompt: "You are a test assistant."
    tools:
      - type: bash
        enabled: true
storage:
  providers:
    - type: memory
"""


@pytest.fixture
def turn_split_config(tmp_path: Path) -> Path:
    """YAML config with TestModel calling bash(sleep 2) for mid-turn steer window."""
    config_path = tmp_path / "turn_split_config.yml"
    config_path.write_text(TURN_SPLIT_CONFIG_YAML.strip() + "\n")
    return config_path


@pytest.fixture
async def subprocess_server_turn_split(
    process_registry: Any,
    turn_split_config: Path,
    allow_model_requests: Any,
) -> Any:
    """Spawn an opencode server with the turn-split config (non-cached)."""
    async for server in _spawn_server(
        "serve-opencode",
        turn_split_config,
        process_registry=process_registry,
        is_stdio=False,
        health_path="/session",
        health_timeout=15.0,
    ):
        yield server


# ---------------------------------------------------------------------------
# Helpers
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


async def _wait_for_messages(
    base_url: str,
    client: httpx.AsyncClient,
    session_id: str,
    expected_count: int,
    *,
    timeout: float = 30.0,
    interval: float = 0.5,
) -> list[dict[str, Any]]:
    """Poll message list until expected_count messages exist or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = await client.get(f"{base_url}/session/{session_id}/message")
        if resp.status_code == 200:
            messages = resp.json()
            if len(messages) >= expected_count:
                return messages
        await asyncio.sleep(interval)
    resp = await client.get(f"{base_url}/session/{session_id}/message")
    return resp.json() if resp.status_code == 200 else []


# ---------------------------------------------------------------------------
# E2E test
# ---------------------------------------------------------------------------


async def test_turn_split_sse_e2e(  # noqa: PLR0915
    subprocess_server_turn_split: SubprocessServer,
    turn_split_config: Path,
) -> None:
    """L4a: Verify logical turn split via SSE when steer arrives mid-turn.

    Verifies that a mid-turn steer message causes a logical turn split,
    producing two assistant messages (A1, A2) with a steer user message
    between them in message ID ordering.

    The test uses both SSE event collection and the messages REST endpoint
    for verification.
    """
    base_url = subprocess_server_turn_split.base_url
    steer_text = "STEER_MARKER_42"

    # Use two clients: one for SSE streaming, one for HTTP requests.
    async with (
        httpx.AsyncClient(timeout=60.0) as sse_client,
        httpx.AsyncClient(timeout=60.0) as http_client,
    ):
        session_id = await _create_session(base_url, http_client)

        # --- Concurrent SSE collection + prompt sending ---
        sse_events: list[dict[str, Any]] = []
        # Track whether we've seen a "busy" status (indicating the run started).
        # Only break on "idle" AFTER seeing "busy" to avoid catching the
        # initial idle state from session creation.
        saw_busy = asyncio.Event()

        async def collect_sse_events() -> None:
            """Collect SSE events until session goes idle (after run) or timeout."""
            deadline = time.monotonic() + 45.0
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

        async def send_prompts() -> None:
            """Send first prompt, wait, then send steer."""
            # Wait for SSE to connect
            await asyncio.sleep(0.5)

            # Send first prompt (non-blocking, starts the run)
            resp = await http_client.post(
                f"{base_url}/session/{session_id}/prompt_async",
                json={"parts": [{"type": "text", "text": "First prompt"}]},
            )
            assert resp.status_code == 204, (
                f"First prompt_async failed: {resp.status_code}: {resp.text}"
            )

            # Wait for the model to respond and bash(sleep 2) to start
            await asyncio.sleep(1.0)

            # Send steer (non-blocking, injects mid-turn)
            steer_resp = await http_client.post(
                f"{base_url}/session/{session_id}/prompt_async",
                json={
                    "parts": [{"type": "text", "text": steer_text}],
                    "delivery": "steer",
                },
            )
            assert steer_resp.status_code == 204, (
                f"Steer prompt_async failed: {steer_resp.status_code}: {steer_resp.text}"
            )

        # Run both tasks concurrently
        sse_task = asyncio.create_task(collect_sse_events(), name="sse_collector")
        prompt_task = asyncio.create_task(send_prompts(), name="prompt_sender")

        try:
            await asyncio.wait_for(asyncio.gather(sse_task, prompt_task), timeout=50.0)
        except TimeoutError:
            pytest.fail(
                f"Timed out waiting for SSE collection. "
                f"Collected {len(sse_events)} events. "
                f"Event types: {[e.get('type') for e in sse_events]}"
            )

        # Also verify via messages endpoint (belt and suspenders)
        messages = await _wait_for_messages(
            base_url, http_client, session_id, expected_count=4, timeout=10.0
        )

    # --- Verification via messages REST endpoint ---
    assistant_messages = [m for m in messages if m.get("info", {}).get("role") == "assistant"]
    user_messages = [m for m in messages if m.get("info", {}).get("role") == "user"]

    assert len(assistant_messages) >= 2, (
        f"Expected at least 2 assistant messages for turn split, "
        f"got {len(assistant_messages)}. "
        f"Message roles: {[m.get('info', {}).get('role') for m in messages]}"
    )
    assert len(user_messages) >= 2, (
        f"Expected at least 2 user messages (first prompt + steer), "
        f"got {len(user_messages)}. "
        f"Message roles: {[m.get('info', {}).get('role') for m in messages]}"
    )

    a1_id = assistant_messages[0]["info"]["id"]
    a2_id = assistant_messages[1]["info"]["id"]
    assert a1_id != a2_id, f"Expected different message IDs for A1 and A2, both are {a1_id}"

    # The steer user message is the second user message (after the first prompt)
    steer_user_id = user_messages[1]["info"]["id"]

    # --- Assertion: Message ID ordering: A1 < steer < A2 ---
    # AgentPool uses ascending IDs (identifier.ascending), so later IDs
    # are lexicographically greater.
    assert a1_id < steer_user_id, (
        f"Expected A1 ID ({a1_id}) < steer user ID ({steer_user_id}). "
        f"This means the steer user message should sort after A1."
    )
    assert steer_user_id < a2_id, (
        f"Expected steer user ID ({steer_user_id}) < A2 ID ({a2_id}). "
        f"This means A2 should sort after the steer user message."
    )

    # --- Verification via SSE events ---
    # Filter message.updated events
    message_updated_infos = [
        info for event in sse_events if (info := _get_message_updated_info(event)) is not None
    ]

    sse_assistant_msgs = [info for info in message_updated_infos if info.get("role") == "assistant"]
    sse_user_msgs = [info for info in message_updated_infos if info.get("role") == "user"]

    # Verify SSE has at least 2 assistant message.updated events
    assert len(sse_assistant_msgs) >= 2, (
        f"Expected at least 2 assistant message.updated events in SSE, "
        f"got {len(sse_assistant_msgs)}. "
        f"SSE event types: {[e.get('type') for e in sse_events]}"
    )

    # Verify SSE has at least 2 user message.updated events
    assert len(sse_user_msgs) >= 2, (
        f"Expected at least 2 user message.updated events in SSE, "
        f"got {len(sse_user_msgs)}. "
        f"SSE event types: {[e.get('type') for e in sse_events]}"
    )

    # Deduplicate assistant messages by ID (replay buffer may send duplicates)
    seen_assistant_ids: list[str] = []
    for info in sse_assistant_msgs:
        msg_id = info["id"]
        if msg_id not in seen_assistant_ids:
            seen_assistant_ids.append(msg_id)

    assert len(seen_assistant_ids) >= 2, (
        f"Expected at least 2 unique assistant message IDs in SSE, "
        f"got {len(seen_assistant_ids)}: {seen_assistant_ids}. "
        f"All assistant infos: {sse_assistant_msgs}"
    )

    sse_a1_id = seen_assistant_ids[0]
    sse_a2_id = seen_assistant_ids[1]
    assert sse_a1_id != sse_a2_id, "Expected different SSE message IDs for A1 and A2"

    # Deduplicate user messages by ID
    seen_user_ids: list[str] = []
    for info in sse_user_msgs:
        msg_id = info["id"]
        if msg_id not in seen_user_ids:
            seen_user_ids.append(msg_id)

    assert len(seen_user_ids) >= 2, (
        f"Expected at least 2 unique user message IDs in SSE, "
        f"got {len(seen_user_ids)}: {seen_user_ids}"
    )

    sse_steer_id = seen_user_ids[1]

    # --- SSE Assertion: Message ID ordering: A1 < steer < A2 ---
    assert sse_a1_id < sse_steer_id, (
        f"Expected SSE A1 ID ({sse_a1_id}) < steer user ID ({sse_steer_id})"
    )
    assert sse_steer_id < sse_a2_id, (
        f"Expected SSE steer user ID ({sse_steer_id}) < A2 ID ({sse_a2_id})"
    )

    # --- SSE Assertion: Event ordering in SSE stream ---
    # Find the first occurrence of each key event in the SSE stream
    a1_index = None
    steer_index = None
    a2_index = None

    for i, event in enumerate(sse_events):
        info = _get_message_updated_info(event)
        if info is None:
            continue
        if info.get("role") == "assistant":
            if a1_index is None and info["id"] == sse_a1_id:
                a1_index = i
            elif a2_index is None and info["id"] == sse_a2_id:
                a2_index = i
        elif info.get("role") == "user" and info["id"] == sse_steer_id:
            steer_index = i

    assert a1_index is not None, "A1 message.updated event not found in SSE stream"
    assert steer_index is not None, "Steer user message.updated event not found in SSE stream"
    assert a2_index is not None, "A2 message.updated event not found in SSE stream"

    assert a1_index < steer_index, (
        f"Expected A1 (index {a1_index}) before steer user message "
        f"(index {steer_index}) in SSE stream"
    )
    assert steer_index < a2_index, (
        f"Expected steer user message (index {steer_index}) before A2 "
        f"(index {a2_index}) in SSE stream"
    )
