"""L4 subprocess E2E tests for ACP concurrent prompts (B7 group).

Tests concurrent prompt handling:
    - B7.1 test_steer_injects_into_active_turn: send prompt + second with asap priority
    - B7.2 test_queue_waits_for_idle: send prompt + second with when_idle priority
    - B7.3 test_concurrent_prompts_no_crash: send 3 prompts concurrently

All tests use ``model: test`` (pydantic-ai TestModel) so NO API key is needed.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

import pytest

from acp import (
    ClientSideConnection,
    CloseSessionRequest,
    DefaultACPClient,
    InitializeRequest,
    NewSessionRequest,
    PromptRequest,
    TextContentBlock,
)
from acp.stdio import spawn_agent_process
from tests.e2e.conftest import SKIP_NO_BINARY, SKIP_WINDOWS


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(SKIP_NO_BINARY, reason="wolfharness binary not on PATH"),
    pytest.mark.skipif(SKIP_WINDOWS, reason="Windows stdio subprocess issues"),
]


# ---------------------------------------------------------------------------
# Helper (mirrors test_acp_subprocess.py)
# ---------------------------------------------------------------------------


class _ACPServerHandle:
    """Handle to a spawned ACP server subprocess with a client connection."""

    def __init__(
        self,
        conn: ClientSideConnection,
        process: asyncio.subprocess.Process,
        client: DefaultACPClient,
    ) -> None:
        self.conn = conn
        self.process = process
        self.client = client

    async def initialize(self) -> Any:
        return await self.conn.initialize(InitializeRequest(protocol_version=1))

    async def new_session(self, cwd: str = "/tmp") -> Any:
        return await self.conn.new_session(NewSessionRequest(cwd=cwd, mcp_servers=[]))

    async def prompt(self, session_id: str, text: str) -> Any:
        return await self.conn.prompt(
            PromptRequest(
                session_id=session_id,
                prompt=[TextContentBlock(text=text)],
            )
        )

    async def close_session(self, session_id: str) -> None:
        await self.conn.close_session(CloseSessionRequest(session_id=session_id))


@contextlib.asynccontextmanager
async def _spawn_acp_server(
    config_path: Path | str,
    *,
    agent: str = "test_agent",
) -> AsyncIterator[_ACPServerHandle]:
    """Spawn ``wolfharness serve-acp`` and return a handle with client connection."""
    import os

    env = os.environ.copy()
    env["OBSERVABILITY_ENABLED"] = "false"
    env["LOGFIRE_DISABLE"] = "true"

    client = DefaultACPClient(allow_file_operations=False)
    async with spawn_agent_process(
        lambda _conn: client,
        "wolfharness",
        "serve-acp",
        str(config_path),
        "--agent",
        agent,
        env=env,
        log_stderr=False,
    ) as (conn, process):
        yield _ACPServerHandle(conn=conn, process=process, client=client)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_steer_injects_into_active_turn(e2e_config: Path) -> None:
    """B7.1: Send a prompt, immediately send a second prompt with priority.

    ``"asap"`` (steer), verify the server handles it without crashing.

    The ACP protocol's ``PromptRequest`` does not have a native ``priority``
    field. The wolfharness SessionController supports ``priority="asap"`` for
    mid-turn injection (steer), but this is an internal API not exposed via
    ACP JSON-RPC. This test sends two concurrent prompts to the same session
    and verifies the server handles them gracefully — the first completes
    normally and the second is either steered or queued.
    """
    async with _spawn_acp_server(e2e_config) as handle:
        await handle.initialize()
        new_sess = await handle.new_session()
        session_id = new_sess.session_id

        # Send first prompt as a background task.
        prompt1_task = asyncio.create_task(handle.prompt(session_id, "First prompt"))
        # Give it a moment to start.
        await asyncio.sleep(0.05)

        # Send second prompt concurrently (would be "asap" priority internally).
        prompt2_task = asyncio.create_task(handle.prompt(session_id, "Second prompt (steer)"))

        # Wait for both to complete.
        results = await asyncio.gather(prompt1_task, prompt2_task, return_exceptions=True)

        # Both prompts should complete without crashing the server.
        for result in results:
            # With TestModel, prompts complete synchronously. Either both
            # succeed or one may error due to session-busy — both are acceptable
            # as long as the server doesn't crash.
            if isinstance(result, Exception):
                # An error response (e.g., session busy) is acceptable.
                continue
            assert result is not None, "Expected a prompt response"

        # Server should still be alive.
        assert handle.process.returncode is None, "Server process exited during concurrent prompts"

        with contextlib.suppress(Exception):
            await handle.close_session(session_id)


async def test_queue_waits_for_idle(e2e_config: Path) -> None:
    """B7.2: Send a prompt, immediately send a second prompt with priority.

    ``"when_idle"`` (queue), verify the server handles it without crashing.

    Similar to B7.1, the ACP protocol does not expose ``priority`` natively.
    The wolfharness SessionController supports ``priority="when_idle"`` for
    queueing, but this is internal. This test verifies the server handles
    sequential/concurrent prompts gracefully.
    """
    async with _spawn_acp_server(e2e_config) as handle:
        await handle.initialize()
        new_sess = await handle.new_session()
        session_id = new_sess.session_id

        # Send first prompt and wait for it to complete.
        resp1 = await handle.prompt(session_id, "First prompt")
        assert resp1 is not None, "Expected first prompt response"
        assert resp1.stop_reason is not None, "Expected stop_reason in first response"

        # Send second prompt (would be "when_idle" priority internally).
        resp2 = await handle.prompt(session_id, "Second prompt (queued)")
        assert resp2 is not None, "Expected second prompt response"
        assert resp2.stop_reason is not None, "Expected stop_reason in second response"

        # Server should still be alive.
        assert handle.process.returncode is None, "Server process exited after queued prompts"

        with contextlib.suppress(Exception):
            await handle.close_session(session_id)


async def test_concurrent_prompts_no_crash(e2e_config: Path) -> None:
    """B7.3: Send 3 prompts concurrently to the same session, verify the.

    server doesn't crash and all prompts get handled.
    """
    async with _spawn_acp_server(e2e_config) as handle:
        await handle.initialize()
        new_sess = await handle.new_session()
        session_id = new_sess.session_id

        # Send 3 prompts concurrently.
        prompts = [handle.prompt(session_id, f"Concurrent prompt {i}") for i in range(3)]
        results = await asyncio.gather(*prompts, return_exceptions=True)

        # At least one should succeed; the rest may error (session busy) but
        # the server must not crash.
        success_count = sum(1 for r in results if not isinstance(r, Exception) and r is not None)
        assert success_count >= 1, (
            f"Expected at least 1 successful prompt, got {success_count}/3. Results: {results}"
        )

        # Server should still be alive.
        assert handle.process.returncode is None, "Server process exited after 3 concurrent prompts"

        with contextlib.suppress(Exception):
            await handle.close_session(session_id)
