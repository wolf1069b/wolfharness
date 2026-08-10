"""L4 subprocess E2E tests for OpenCode server lock concurrency.

Tests that the per-session lock in the OpenCode server does not block
concurrent operations (prompt_async, session close) while a synchronous
POST /message is waiting for the agent to complete.

Root cause: _process_message() acquires state.get_session_lock() and holds
it through wait_for_completion(), blocking all other endpoints that need
the same lock.

Fix: Split into lock-held (load + create + route) + lock-free (wait +
finalize) + lock-held (mark idle).

Run:  uv run pytest tests/e2e/test_opencode_lock_concurrency.py -v -m "e2e"
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
# Config: model with long delay to keep sync POST /message holding the lock
# ---------------------------------------------------------------------------

LONG_DELAY_MODEL_CONFIG = """

agents:
  test_agent:
    type: native
    model:
      type: test
      custom_output_text: "Response after delay"
      pre_stream_delay: 10.0
    system_prompt: "You are a test assistant."
storage:
  providers:
    - type: memory
"""


@pytest.fixture
def long_delay_config(tmp_path: Path) -> Path:
    """YAML config with a 10s-delay test model.

    The 10s delay ensures the sync POST /message holds the lock long
    enough for concurrent requests to contend. If the lock is held
    during wait_for_completion, prompt_async will block for ~10s.
    """
    config_path = tmp_path / "long_delay_config.yml"
    config_path.write_text(LONG_DELAY_MODEL_CONFIG.strip() + "\n")
    return config_path


@pytest.fixture
async def subprocess_server_long_delay(
    process_registry: Any,
    long_delay_config: Path,
    allow_model_requests: Any,
) -> Any:
    """Spawn an opencode server with the 10s-delay test model config."""
    async for server in _spawn_server(
        "serve-opencode",
        long_delay_config,
        process_registry=process_registry,
        is_stdio=False,
        health_path="/session",
        health_timeout=15.0,
    ):
        yield server


# ---------------------------------------------------------------------------
# Config: fast model for tests that need quick completion
# ---------------------------------------------------------------------------

FAST_MODEL_CONFIG = """

agents:
  test_agent:
    type: native
    model:
      type: test
      custom_output_text: "Fast response"
    system_prompt: "You are a test assistant."
storage:
  providers:
    - type: memory
"""


@pytest.fixture
def fast_config(tmp_path: Path) -> Path:
    """YAML config with a fast test model (no delay)."""
    config_path = tmp_path / "fast_config.yml"
    config_path.write_text(FAST_MODEL_CONFIG.strip() + "\n")
    return config_path


@pytest.fixture
async def subprocess_server_fast(
    process_registry: Any,
    fast_config: Path,
    allow_model_requests: Any,
) -> Any:
    """Spawn an opencode server with the fast test model config."""
    async for server in _spawn_server(
        "serve-opencode",
        fast_config,
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


# ---------------------------------------------------------------------------
# Tests: Lock split — prompt_async not blocked by sync
# ---------------------------------------------------------------------------


class TestPromptAsyncNotBlockedBySync:
    """prompt_async must not be blocked by a running sync POST /message."""

    async def test_prompt_async_returns_quickly_while_sync_running(
        self,
        subprocess_server_long_delay: SubprocessServer,
    ) -> None:
        """While sync POST /message is running (10s model), prompt_async.

        must return 204 within 2s — not wait for the sync message to complete.

        Given: a session with a 10s-delay model
        When: sync POST /message is sent (starts 10s agent run)
        And:  prompt_async is sent 0.5s later
        Then: prompt_async returns 204 in < 2s (not blocked by lock)
        """
        base_url = subprocess_server_long_delay.base_url

        async with httpx.AsyncClient(timeout=60.0) as client:
            session_id = await _create_session(base_url, client)

            # Start sync message (blocks for ~10s with long-delay model)
            sync_task = asyncio.create_task(
                _send_message_sync(base_url, client, session_id, "sync message")
            )

            try:
                # Give sync time to acquire lock and enter wait_for_completion
                await asyncio.sleep(0.5)

                # Send prompt_async while sync is holding the lock
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
                assert elapsed < 2.0, (
                    f"prompt_async took {elapsed:.2f}s — blocked by sync message "
                    f"lock held during wait_for_completion. Should return < 2s "
                    f"even while agent is running (10s model)."
                )
            finally:
                if not sync_task.done():
                    sync_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await sync_task
                else:
                    sync_resp = await sync_task
                    assert sync_resp.status_code in (200, 201)

    async def test_prompt_async_user_message_created_while_sync_running(
        self,
        subprocess_server_long_delay: SubprocessServer,
    ) -> None:
        """prompt_async's user message must appear in the session while.

        sync POST /message is still running.

        Given: a session with a 10s-delay model
        When: sync POST /message is sent (starts 10s agent run)
        And:  prompt_async is sent 0.5s later
        Then: the async user message appears in GET /message within 3s
              (proving it was created and routed, not just 204'd)
        """
        base_url = subprocess_server_long_delay.base_url

        async with httpx.AsyncClient(timeout=60.0) as client:
            session_id = await _create_session(base_url, client)

            sync_task = asyncio.create_task(
                _send_message_sync(base_url, client, session_id, "sync message")
            )

            try:
                await asyncio.sleep(0.5)

                resp = await _send_prompt_async(
                    base_url,
                    client,
                    session_id,
                    "async inject",
                    delivery="queue",
                )
                assert resp.status_code == 204

                # The async user message should appear in the session
                # even while sync is still running
                messages = await _wait_for_message_count(
                    base_url, client, session_id, 2, timeout=3.0
                )
                assert len(messages) >= 2, (
                    f"Expected >= 2 messages (sync user + async user) within 3s, "
                    f"got {len(messages)}. The async user message was not created "
                    f"because prompt_async was blocked by the sync lock."
                )
            finally:
                if not sync_task.done():
                    sync_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await sync_task
                else:
                    await sync_task


# ---------------------------------------------------------------------------
# Tests: Finalization correctness after lock split
# ---------------------------------------------------------------------------


class TestFinalizationAfterLockSplit:
    """Finalization (time.completed, message persistence) must work.

    correctly when wait_for_completion runs outside the lock.
    """

    async def test_sync_message_sets_time_completed(
        self,
        subprocess_server_fast: SubprocessServer,
    ) -> None:
        """After POST /message, the assistant message must have time.completed set.

        In the async model, POST /message returns immediately with
        time.completed = None. The event consumer sets time.completed
        on StreamCompleteEvent. We poll GET /message to verify.

        Given: a session with a fast model
        When: POST /message is sent
        Then: GET /message shows assistant message with time.completed != None
        """
        base_url = subprocess_server_fast.base_url

        async with httpx.AsyncClient(timeout=30.0) as client:
            session_id = await _create_session(base_url, client)

            resp = await _send_message_sync(base_url, client, session_id, "hello")
            assert resp.status_code in (200, 201)

            # In the async model, time.completed is None in the initial
            # response. Poll GET /messages until the event consumer finalizes.
            messages = await _wait_for_message_count(base_url, client, session_id, 2, timeout=10.0)
            assert len(messages) >= 2

            # Poll until time.completed is set (finalization may lag behind message creation)
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                assistant = messages[1]
                time_field = assistant.get("info", {}).get("time", {})
                if time_field.get("completed") is not None:
                    break
                await asyncio.sleep(0.5)
                messages = await _wait_for_message_count(
                    base_url, client, session_id, 2, timeout=5.0
                )

            assistant = messages[1]
            time_field = assistant.get("info", {}).get("time", {})
            assert time_field.get("completed") is not None, (
                "time.completed not set on assistant message — finalization "
                "may have been skipped after lock split."
            )

    async def test_sync_message_persists_to_storage(
        self,
        subprocess_server_fast: SubprocessServer,
    ) -> None:
        """After sync POST /message completes, the assistant message must.

        be retrievable via GET /message — proving persistence ran lock-free.

        Given: a session with a fast model
        When: sync POST /message is sent and completes
        Then: GET /message returns both user and assistant messages
        """
        base_url = subprocess_server_fast.base_url

        async with httpx.AsyncClient(timeout=30.0) as client:
            session_id = await _create_session(base_url, client)

            await _send_message_sync(base_url, client, session_id, "hello")

            messages = await _wait_for_message_count(base_url, client, session_id, 2, timeout=5.0)
            assert len(messages) >= 2, (
                f"Expected >= 2 messages after sync completion, got {len(messages)}"
            )

            roles = [m.get("info", {}).get("role") for m in messages[:2]]
            assert roles == ["user", "assistant"], (
                f"Expected [user, assistant] ordering, got {roles}"
            )

            # Verify assistant has completed time
            # (finalization may lag behind message creation; poll if needed)
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                assistant = messages[1]
                time_field = assistant.get("info", {}).get("time", {})
                if time_field.get("completed") is not None:
                    break
                await asyncio.sleep(0.5)
                messages = await _wait_for_message_count(
                    base_url, client, session_id, 2, timeout=5.0
                )

            assistant = messages[1]
            time_field = assistant.get("info", {}).get("time", {})
            assert time_field.get("completed") is not None, (
                "Assistant message missing time.completed — persistence or "
                "finalization may have failed after lock split."
            )


# ---------------------------------------------------------------------------
# Tests: Session close during lock-free wait
# ---------------------------------------------------------------------------


class TestSessionCloseDuringSyncRun:
    """Closing a session while sync POST /message is in the lock-free wait.

    phase must not deadlock or crash.
    """

    async def test_close_session_while_sync_running_no_deadlock(
        self,
        subprocess_server_long_delay: SubprocessServer,
    ) -> None:
        """DELETE /session while sync POST /message is running must.

        complete without deadlock.

        Given: a session with a 10s-delay model
        When: sync POST /message is sent (starts 10s agent run)
        And:  DELETE /session is sent 0.5s later
        Then: DELETE completes within 5s (no deadlock)
        And:  sync task completes (cancelled or error) within 5s
        """
        base_url = subprocess_server_long_delay.base_url

        async with httpx.AsyncClient(timeout=60.0) as client:
            session_id = await _create_session(base_url, client)

            sync_task = asyncio.create_task(
                _send_message_sync(base_url, client, session_id, "sync message")
            )

            # Wait for sync to enter wait_for_completion
            await asyncio.sleep(0.5)

            # Close the session — should not deadlock
            start = time.monotonic()
            resp = await client.delete(f"{base_url}/session/{session_id}")
            elapsed = time.monotonic() - start

            assert resp.status_code in (200, 204), f"DELETE /session returned {resp.status_code}"
            assert elapsed < 5.0, (
                f"DELETE /session took {elapsed:.2f}s — possible deadlock with sync message lock."
            )

            # Sync task should complete (not hang)
            try:
                await asyncio.wait_for(sync_task, timeout=5.0)
            except TimeoutError:
                pytest.fail(
                    "sync POST /message did not complete within 5s after "
                    "session close — possible deadlock."
                )
            except asyncio.CancelledError:
                pass  # Acceptable — request was cancelled
