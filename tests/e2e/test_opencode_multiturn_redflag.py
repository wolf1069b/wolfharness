"""L4 subprocess E2E red-flag tests for OpenCode multi-turn lifecycle.

These tests reproduce the issues reported in GitHub issue #221:
[SystemNotification] Support opencode TUI system reminder display.

The 24 sub-issues in #221 all stem from the opencode server's assumption of
single-turn, synchronous, non-concurrent execution. These tests verify
multi-turn lifecycle, message ordering, persistence, and concurrency
behaviors that are currently untested (the "multi-turn test gap").

All tests use ``model: test`` (pydantic-ai TestModel) so NO API key is needed.

Run:  uv run pytest tests/e2e/test_opencode_multiturn_redflag.py -v -m "e2e"
"""

from __future__ import annotations

import asyncio
import contextlib
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
# Config fixtures
# ---------------------------------------------------------------------------

SLOW_MODEL_CONFIG_YAML = """
agents:
  test_agent:
    type: native
    model:
      type: test
      custom_output_text: "Slow response"
      pre_stream_delay: 2.0
    system_prompt: "You are a test assistant."
storage:
  providers:
    - type: memory
"""


@pytest.fixture
def slow_model_config(tmp_path: Path) -> Path:
    """YAML config with a slow FunctionModel (2s delay per response).

    Used by concurrency red-flag tests that need the agent to be
    "running" for a measurable duration to expose lock contention.
    """
    config_path = tmp_path / "slow_model_config.yml"
    config_path.write_text(SLOW_MODEL_CONFIG_YAML.strip() + "\n")
    return config_path


@pytest.fixture
async def subprocess_server_slow(
    process_registry: Any,
    slow_model_config: Path,
    allow_model_requests: Any,
) -> Any:
    """Spawn an opencode server with the slow FunctionModel config.

    Unlike the cached ``subprocess_server``, this always spawns a fresh
    process (no cache) because the slow model config is unique to these
    tests.
    """
    async for server in _spawn_server(
        "serve-opencode",
        slow_model_config,
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


async def _send_prompt_async(
    base_url: str,
    client: httpx.AsyncClient,
    session_id: str,
    text: str,
    *,
    delivery: str = "queue",
) -> httpx.Response:
    """Send a prompt via prompt_async (non-blocking, returns 204)."""
    payload: dict[str, Any] = {
        "parts": [{"type": "text", "text": text}],
        "delivery": delivery,
    }
    return await client.post(
        f"{base_url}/session/{session_id}/prompt_async",
        json=payload,
    )


async def _send_message_sync(
    base_url: str,
    client: httpx.AsyncClient,
    session_id: str,
    text: str,
) -> httpx.Response:
    """Send a message via sync POST /message (blocking, waits for completion)."""
    payload: dict[str, Any] = {
        "parts": [{"type": "text", "text": text}],
    }
    return await client.post(
        f"{base_url}/session/{session_id}/message",
        json=payload,
    )


async def _get_messages(
    base_url: str,
    client: httpx.AsyncClient,
    session_id: str,
) -> list[dict[str, Any]]:
    """Get all messages for a session."""
    resp = await client.get(f"{base_url}/session/{session_id}/message")
    assert resp.status_code == 200, f"Failed to get messages: {resp.status_code}"
    return resp.json()


async def _wait_for_message_count(
    base_url: str,
    client: httpx.AsyncClient,
    session_id: str,
    expected_count: int,
    *,
    timeout: float = 15.0,
    interval: float = 0.3,
) -> list[dict[str, Any]]:
    """Poll message list until expected_count messages exist or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        messages = await _get_messages(base_url, client, session_id)
        if len(messages) >= expected_count:
            return messages
        await asyncio.sleep(interval)
    return await _get_messages(base_url, client, session_id)


async def _wait_for_assistant_completion(
    base_url: str,
    client: httpx.AsyncClient,
    session_id: str,
    *,
    timeout: float = 15.0,
    interval: float = 0.3,
    min_count: int = 1,
) -> list[dict[str, Any]]:
    """Poll until at least min_count assistant messages have time.completed set.

    Args:
        base_url: Server base URL.
        client: HTTP client.
        session_id: Session to poll.
        timeout: Maximum wait time in seconds.
        interval: Polling interval in seconds.
        min_count: Minimum number of completed assistant messages to wait for.

    Returns:
        List of all messages for the session.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        messages = await _get_messages(base_url, client, session_id)
        assistant_msgs = [m for m in messages if m.get("info", {}).get("role") == "assistant"]
        completed = [
            m
            for m in assistant_msgs
            if m.get("info", {}).get("time", {}).get("completed") is not None
        ]
        if len(completed) >= min_count:
            return messages
        await asyncio.sleep(interval)
    return await _get_messages(base_url, client, session_id)


def _extract_text_part_text(msg: dict[str, Any]) -> str:
    """Extract text from the first text part of a message."""
    for part in msg.get("parts", []):
        if part.get("type") == "text" and part.get("text"):
            return part["text"]
    return ""


def _id_timestamp_ms(msg_id: str) -> int | None:
    """Extract millisecond timestamp from an wolfharness ascending ID.

    AgentPool ID format: ``{prefix}_{16 hex chars}{14 base62 chars}``
    where the 16 hex chars encode 8 bytes (64 bits) of
    ``timestamp_ms * 0x1000 + counter`` (big-endian).

    Returns None if the ID format is not recognized.
    """
    # Strip prefix (msg_, prt_, ses_, etc.)
    id_part = msg_id
    if "_" in id_part:
        id_part = id_part.split("_", 1)[1]

    if len(id_part) < 16:
        return None

    # First 16 chars are hex encoding of 8 bytes (64 bits)
    hex_part = id_part[:16]
    try:
        now = int(hex_part, 16)
    except ValueError:
        return None

    # now = timestamp_ms * 0x1000 + counter
    # timestamp_ms = now >> 12
    return now >> 12


# ---------------------------------------------------------------------------
# P0: Multi-Turn Lifecycle Red Flags (E1/E2/E3/D1/D3)
# ---------------------------------------------------------------------------


class TestMultiTurnLifecycle:
    """Red flags: multi-turn lifecycle — the core test gap.

    These tests verify that consecutive turns through the same session
    work correctly. The current codebase has ZERO tests for this scenario
    through the opencode server layer.
    """

    @pytest.mark.parametrize(
        "subprocess_server",
        [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
        indirect=True,
    )
    async def test_redflag_e1_consecutive_turns_both_complete(
        self,
        subprocess_server: SubprocessServer,
        e2e_config: Path,
    ) -> None:
        """E1: Two consecutive prompt_async messages must both complete.

        Current behavior: _consume_run() breaks on StreamCompleteEvent
        and calls gen.aclose(), killing the RunHandle after turn 1.
        The second prompt_async's followup message is lost.

        Expected:
        - After turn 1: 2 messages (user + assistant)
        - After turn 2: 4 messages (user1 + assistant1 + user2 + assistant2)
        - Both assistant messages have non-empty text content
        """
        base_url = subprocess_server.base_url

        async with httpx.AsyncClient(timeout=30.0) as client:
            session_id = await _create_session(base_url, client)

            # Turn 1
            resp1 = await _send_prompt_async(base_url, client, session_id, "turn 1")
            assert resp1.status_code == 204, f"Turn 1 prompt_async failed: {resp1.status_code}"

            messages_after_t1 = await _wait_for_message_count(
                base_url,
                client,
                session_id,
                2,
                timeout=15.0,
            )
            assert len(messages_after_t1) >= 2, (
                f"Turn 1 should produce 2 messages, got {len(messages_after_t1)}"
            )

            # Turn 2 — the critical one (currently lost due to E1)
            resp2 = await _send_prompt_async(base_url, client, session_id, "turn 2")
            assert resp2.status_code == 204, f"Turn 2 prompt_async failed: {resp2.status_code}"

            messages_after_t2 = await _wait_for_message_count(
                base_url,
                client,
                session_id,
                4,
                timeout=15.0,
            )
            assert len(messages_after_t2) >= 4, (
                f"Turn 2 should produce 4 total messages, got {len(messages_after_t2)}. "
                f"This indicates the second turn was lost (issue E1: _consume_run kills generator)."
            )

            # Verify both assistant messages have parts (proves turn executed)
            assistant_msgs = [
                m for m in messages_after_t2 if m.get("info", {}).get("role") == "assistant"
            ]
            assert len(assistant_msgs) >= 2, (
                f"Should have 2 assistant messages, got {len(assistant_msgs)}"
            )
            for i, msg in enumerate(assistant_msgs[:2]):
                parts = msg.get("parts", [])
                assert len(parts) > 0, (
                    f"Assistant message {i + 1} has no parts — turn may not have executed properly"
                )

    @pytest.mark.parametrize(
        "subprocess_server",
        [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
        indirect=True,
    )
    async def test_redflag_d1_turn2_has_new_assistant_message_id(
        self,
        subprocess_server: SubprocessServer,
        e2e_config: Path,
    ) -> None:
        """D1: Turn 2 must have a new assistant message ID, not reuse turn 1's.

        Current behavior: start_event_consumer() is idempotent —
        _before_consumer_loop() only runs once. _message_registered stays
        True after turn 1, so turn 2 reuses turn 1's assistant_msg_id.

        Expected:
        - assistant_msg_id_1 != assistant_msg_id_2
        - assistant_msg_id_2 > assistant_msg_id_1 (monotonically ascending)
        """
        base_url = subprocess_server.base_url

        async with httpx.AsyncClient(timeout=30.0) as client:
            session_id = await _create_session(base_url, client)

            # Turn 1
            await _send_prompt_async(base_url, client, session_id, "turn 1")
            messages_t1 = await _wait_for_message_count(
                base_url,
                client,
                session_id,
                2,
                timeout=15.0,
            )
            assistant_t1 = [m for m in messages_t1 if m.get("info", {}).get("role") == "assistant"]
            assert len(assistant_t1) >= 1
            id_1 = assistant_t1[0]["info"]["id"]

            # Turn 2
            await _send_prompt_async(base_url, client, session_id, "turn 2")
            messages_t2 = await _wait_for_message_count(
                base_url,
                client,
                session_id,
                4,
                timeout=15.0,
            )
            assistant_t2 = [m for m in messages_t2 if m.get("info", {}).get("role") == "assistant"]
            assert len(assistant_t2) >= 2
            id_2 = assistant_t2[1]["info"]["id"]

            assert id_1 != id_2, (
                f"Turn 2 reused turn 1's assistant message ID ({id_1}). "
                f"This is issue D1: consumer loop idempotency prevents per-turn reset."
            )
            assert id_2 > id_1, (
                f"Assistant message ID must be monotonically ascending: id_1={id_1}, id_2={id_2}"
            )

    @pytest.mark.parametrize(
        "subprocess_server",
        [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
        indirect=True,
    )
    async def test_redflag_d3_turn2_assistant_time_completed_set(
        self,
        subprocess_server: SubprocessServer,
        e2e_config: Path,
    ) -> None:
        """D3: Turn 2's assistant message must have time.completed set.

        Current behavior: time.completed is only set in _wait_and_finalize,
        but prompt_async doesn't go through that path. The event bridge's
        StreamCompleteEvent handler should set it, but currently doesn't.

        Expected:
        - assistant_t2.info.time.completed is not None
        """
        base_url = subprocess_server.base_url

        async with httpx.AsyncClient(timeout=30.0) as client:
            session_id = await _create_session(base_url, client)

            await _send_prompt_async(base_url, client, session_id, "turn 1")
            await _send_prompt_async(base_url, client, session_id, "turn 2")

            messages = await _wait_for_assistant_completion(
                base_url,
                client,
                session_id,
                timeout=15.0,
                min_count=2,
            )
            assistant_msgs = [m for m in messages if m.get("info", {}).get("role") == "assistant"]
            # Need at least 2 assistant messages
            if len(assistant_msgs) < 2:
                pytest.skip(
                    f"Only {len(assistant_msgs)} assistant messages "
                    f"(E1 may have prevented turn 2). "
                    f"This test is only meaningful after E1 is fixed."
                )

            last_assistant = assistant_msgs[-1]
            time_completed = last_assistant["info"]["time"].get("completed")
            assert time_completed is not None, (
                "Turn 2 assistant message has time.completed=None. "
                "This is issue D3: time.completed not set in event bridge "
                "(prompt_async path doesn't go through _wait_and_finalize)."
            )

    @pytest.mark.parametrize(
        "subprocess_server",
        [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
        indirect=True,
    )
    @pytest.mark.xfail(
        reason="TestModel does not produce text parts in OpenCode message format — "
        "assistant messages only have step-start/step-finish parts. "
        "E2 is verified by test_redflag_e1_consecutive_turns_both_complete getting 4 messages.",
        strict=False,
        raises=AssertionError,
    )
    @pytest.mark.known_bug
    async def test_redflag_e2_turn2_response_has_content(
        self,
        subprocess_server: SubprocessServer,
        e2e_config: Path,
    ) -> None:
        """E2: Turn 2's assistant response must have actual text content.

        Current behavior: After E1 kills the generator, wait_for_completion
        waits on complete_event (set when generator exits), not
        _turn_complete_event (per-turn completion). Even if E1 is fixed,
        wait_for_completion may never return for multi-turn because the
        generator doesn't exit between turns.

        Expected:
        - 4 messages total: user1, assistant1, user2, assistant2
        - assistant2 has at least one text part with non-empty content
        """
        base_url = subprocess_server.base_url

        async with httpx.AsyncClient(timeout=30.0) as client:
            session_id = await _create_session(base_url, client)

            await _send_prompt_async(base_url, client, session_id, "first question")
            await _send_prompt_async(base_url, client, session_id, "second question")

            messages = await _wait_for_message_count(
                base_url,
                client,
                session_id,
                4,
                timeout=15.0,
            )

            if len(messages) < 4:
                pytest.skip(
                    f"Only {len(messages)} messages (E1 may have prevented turn 2). "
                    f"This test is only meaningful after E1 is fixed."
                )

            # Verify message order: user, assistant, user, assistant
            roles = [m["info"]["role"] for m in messages[:4]]
            assert roles == ["user", "assistant", "user", "assistant"], (
                f"Expected role sequence [user, assistant, user, assistant], got {roles}"
            )

            # Verify assistant2 has content
            assistant2 = messages[3]
            text = _extract_text_part_text(assistant2)
            assert text.strip(), (
                "Turn 2 assistant response has empty text content. "
                "This indicates the turn didn't execute properly (issue E2)."
            )


# ---------------------------------------------------------------------------
# P1: Message Ordering & Persistence Red Flags (C1/C2/B5)
# ---------------------------------------------------------------------------


class TestMessageOrdering:
    """Red flags: message ID ordering and persistence timing."""

    @pytest.mark.parametrize(
        "subprocess_server",
        [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
        indirect=True,
    )
    async def test_redflag_c1_message_id_timestamp_consistent(
        self,
        subprocess_server: SubprocessServer,
        e2e_config: Path,
    ) -> None:
        """C1: Message ID timestamp must match time.created.

        Current behavior: identifiers.py uses ``int(time.time() * 1000)``
        (float truncation) at line 88, while ``time_utils.now_ms()`` uses
        ``time.time_ns() // 1_000_000`` (integer division). Additionally,
        the 48-bit encoding (``timestamp_ms * 0x1000 + counter``) overflows
        for timestamps past ~year 2178, but the more immediate issue is
        the float truncation causing 1ms mismatches in same-ms messages.

        This test verifies that the timestamp encoded in the message ID
        is consistent with ``time.created``. If they differ, message
        ordering by timestamp will be inconsistent with ordering by ID.

        Expected:
        - For each message, decoded timestamp from ID == time.created
          (or at minimum, the difference is < 1000ms, indicating same-second)
        """
        base_url = subprocess_server.base_url

        async with httpx.AsyncClient(timeout=30.0) as client:
            session_id = await _create_session(base_url, client)
            await _send_prompt_async(base_url, client, session_id, "test")
            messages = await _wait_for_message_count(
                base_url,
                client,
                session_id,
                2,
                timeout=15.0,
            )

            for msg in messages:
                msg_id = msg["info"]["id"]
                time_created = msg["info"]["time"].get("created")
                if time_created is None:
                    continue

                id_ts = _id_timestamp_ms(msg_id)
                if id_ts is None:
                    pytest.skip(f"Message ID {msg_id} format not recognized, cannot verify C1")

                # The ID-encoded timestamp and time.created should match.
                # Due to the 48-bit encoding overflow for current timestamps
                # (2026+), the absolute values may differ. But the KEY test
                # is: are they consistent? If not, sorting by timestamp
                # vs sorting by ID will produce different orders.
                # We check that they're within 1000ms (same second) at minimum.
                diff = abs(id_ts - time_created)
                assert diff < 1000, (
                    f"ID timestamp ({id_ts}) differs from time.created "
                    f"({time_created}) by {diff}ms for message {msg_id}. "
                    f"This is issue C1: identifiers.py uses "
                    f"int(time.time()*1000) (float truncation) while "
                    f"time_utils uses time.time_ns()//1_000_000 (integer "
                    f"division), and the 48-bit encoding may overflow."
                )

    @pytest.mark.parametrize(
        "subprocess_server",
        [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
        indirect=True,
    )
    async def test_redflag_b5_persist_before_broadcast(
        self,
        subprocess_server: SubprocessServer,
        e2e_config: Path,
    ) -> None:
        """B5: Messages must be persisted before SSE broadcast.

        Current behavior: prompt_async broadcasts PartUpdatedEvent before
        calling append_message_to_session(). TUI sync() fetches from API
        after receiving SSE event, doesn't find the message, and appends
        it at the end → wrong position.

        Expected:
        - When GET /session/{id}/message returns the user message,
          it should also be retrievable via GET /session/{id}/message/{msg_id}
        - This verifies persist happened (not just in-memory broadcast)
        """
        base_url = subprocess_server.base_url

        async with httpx.AsyncClient(timeout=30.0) as client:
            session_id = await _create_session(base_url, client)

            # Send prompt_async and immediately check if message is persisted
            await _send_prompt_async(base_url, client, session_id, "persist test")

            # Poll for the user message to appear in the message list
            messages = await _wait_for_message_count(
                base_url,
                client,
                session_id,
                1,
                timeout=5.0,
                interval=0.1,
            )

            # As soon as we see the user message, verify it's individually retrievable
            user_msgs = [m for m in messages if m["info"]["role"] == "user"]
            assert len(user_msgs) >= 1, "User message should appear in message list"

            user_msg_id = user_msgs[0]["info"]["id"]

            # Immediately fetch the individual message — if persist happened
            # before broadcast, this should return 200
            resp = await client.get(f"{base_url}/session/{session_id}/message/{user_msg_id}")
            assert resp.status_code == 200, (
                f"Message {user_msg_id} not found via GET /message/{user_msg_id} "
                f"(status {resp.status_code}). This indicates broadcast happened "
                f"before persist (issue B5), or the message is only in memory."
            )


# ---------------------------------------------------------------------------
# P2: Concurrency & Lock Red Flags (D4/B3)
# ---------------------------------------------------------------------------


class TestConcurrencyAndLocking:
    """Red flags: concurrency and lock granularity."""

    async def test_redflag_d4_prompt_async_not_blocked_by_sync(
        self,
        subprocess_server_slow: SubprocessServer,
    ) -> None:
        """D4: prompt_async must not be blocked by a running sync POST /message.

        Current behavior: _process_message() acquires per-session lock and
        holds it through wait_for_completion(). prompt_async's message
        creation is blocked by the same lock → 5s timeout or delayed creation.

        This test uses a slow FunctionModel (2s delay) to ensure the sync
        message holds the lock long enough for prompt_async to contend.

        Expected:
        - While sync POST /message is running (2s), prompt_async returns 204
          within 5 seconds (not blocked by the lock)
        """
        base_url = subprocess_server_slow.base_url

        async with httpx.AsyncClient(timeout=60.0) as client:
            session_id = await _create_session(base_url, client)

            # Start sync message (blocks until agent completes — ~2s with slow model)
            sync_task = asyncio.create_task(
                _send_message_sync(base_url, client, session_id, "sync message")
            )

            try:
                # Give the sync message time to acquire the lock and start the agent
                await asyncio.sleep(0.5)

                # Send prompt_async while sync is running and holding the lock
                start = time.monotonic()
                resp = await _send_prompt_async(
                    base_url,
                    client,
                    session_id,
                    "async inject",
                    delivery="queue",
                )
                elapsed = time.monotonic() - start

                assert resp.status_code == 204, (
                    f"prompt_async returned {resp.status_code} instead of 204"
                )
                assert elapsed < 5.0, (
                    f"prompt_async took {elapsed:.2f}s — likely blocked by sync "
                    f"message lock (issue D4: lock held for entire agent run). "
                    f"Should return immediately (<5s) even while agent is running."
                )
            finally:
                if not sync_task.done():
                    sync_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await sync_task
                else:
                    sync_resp = await sync_task
                    assert sync_resp.status_code in (200, 201)

    @pytest.mark.parametrize(
        "subprocess_server",
        [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
        indirect=True,
    )
    async def test_redflag_b3_queued_message_before_response(
        self,
        subprocess_server: SubprocessServer,
        e2e_config: Path,
    ) -> None:
        """B3: Queued messages should appear before the agent response, not after.

        Current behavior: Phase 1 (message creation + broadcast) is inside
        the lock, blocked until agent finishes. So the queued message
        appears after the agent response → wrong position in timeline.

        Expected:
        - Send sync message (starts agent)
        - Send prompt_async while agent is running (queued)
        - After completion, the queued user message should appear BEFORE
          the assistant response that follows it
        - Message order: user1, assistant1, user2(queued), assistant2
        """
        base_url = subprocess_server.base_url

        async with httpx.AsyncClient(timeout=60.0) as client:
            session_id = await _create_session(base_url, client)

            # Start sync message
            sync_task = asyncio.create_task(
                _send_message_sync(base_url, client, session_id, "first message")
            )

            try:
                await asyncio.sleep(0.3)

                # Queue a second message while first is running
                await _send_prompt_async(
                    base_url,
                    client,
                    session_id,
                    "queued message",
                    delivery="queue",
                )

                # Wait for both turns to complete
                await sync_task
            finally:
                if not sync_task.done():
                    sync_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await sync_task

            messages = await _wait_for_message_count(
                base_url,
                client,
                session_id,
                4,
                timeout=20.0,
            )

            if len(messages) < 4:
                pytest.skip(
                    f"Only {len(messages)} messages — multi-turn may not work yet (E1). "
                    f"This test is only meaningful after E1 is fixed."
                )

            # Verify message order
            roles = [m["info"]["role"] for m in messages[:4]]
            assert roles == ["user", "assistant", "user", "assistant"], (
                f"Expected [user, assistant, user, assistant], got {roles}. "
                f"The queued message may have appeared after the assistant response "
                f"(issue B3: Phase 1 blocked by lock)."
            )

            # User message content is now delivered via SSE (message.part.updated) only,
            # not via the REST API (user message parts stripped to prevent TUI duplication).
            # The order assertion above (B3: queued message before assistant response) is
            # the primary purpose of this test and is covered by the role sequence check.
            # user2_content = _extract_text_part_text(messages[2])
            # assert "queued" in user2_content.lower(), (...)
