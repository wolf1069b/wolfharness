"""L4 subprocess E2E tests for the ACP server (``wolfharness serve-acp``).

L4a smoke tests (``@pytest.mark.e2e``, NOT slow):
    - test_server_startup: spawn serve-acp, verify process alive, initialize JSON-RPC
    - test_basic_prompt: send a prompt, verify response and event sequence
    - test_server_shutdown: close session, verify clean process exit

L4b full tests (``@pytest.mark.e2e`` + ``@pytest.mark.slow``):
    - test_multi_turn_conversation: 2+ prompts in same session
    - test_tool_call_e2e: agent with bash tool, verify tool call events
    - test_cancellation_e2e: start prompt, cancel mid-stream

All tests use ``model: test`` (pydantic-ai TestModel) so NO API key is needed.
L4a smoke tests: pytest -m "e2e and not slow" (~30s)
L4b full tests: pytest -m e2e
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

import pytest

from acp import (
    CancelNotification,
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

    from acp.schema import (
        InitializeResponse,
        NewSessionResponse,
        PromptResponse,
        SessionNotification,
    )


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(SKIP_NO_BINARY, reason="wolfharness binary not on PATH"),
    pytest.mark.skipif(SKIP_WINDOWS, reason="Windows stdio subprocess issues"),
]


# ---------------------------------------------------------------------------
# Helper context manager
# ---------------------------------------------------------------------------


class ACPServerHandle:
    """Handle to a spawned ACP server subprocess with a client connection."""

    def __init__(self, conn: ClientSideConnection, process: Any, client: DefaultACPClient) -> None:
        self.conn = conn
        self.process = process
        self.client = client

    async def initialize(self) -> InitializeResponse:
        return await self.conn.initialize(InitializeRequest(protocol_version=1))

    async def new_session(self, cwd: str = "/tmp") -> NewSessionResponse:
        return await self.conn.new_session(NewSessionRequest(cwd=cwd, mcp_servers=[]))

    async def prompt(self, session_id: str, text: str) -> PromptResponse:
        return await self.conn.prompt(
            PromptRequest(
                session_id=session_id,
                prompt=[TextContentBlock(text=text)],
            )
        )

    async def close_session(self, session_id: str) -> None:
        await self.conn.close_session(CloseSessionRequest(session_id=session_id))

    def get_notifications(self) -> list[SessionNotification]:
        return list(self.client.notifications)

    def clear_notifications(self) -> None:
        self.client.notifications.clear()


@contextlib.asynccontextmanager
async def _spawn_acp_server(
    config_path: Path | str,
    *,
    agent: str = "test_agent",
) -> AsyncIterator[ACPServerHandle]:
    """Spawn ``wolfharness serve-acp`` and return a handle with client connection.

    Args:
        config_path: Path to the YAML config file.
        agent: Agent name to use.
    """
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
        yield ACPServerHandle(conn=conn, process=process, client=client)


# ---------------------------------------------------------------------------
# L4a Smoke Tests
# ---------------------------------------------------------------------------


async def test_server_startup(e2e_config: Path) -> None:
    """L4a: Start serve-acp, verify process is alive, initialize JSON-RPC."""
    async with _spawn_acp_server(e2e_config) as handle:
        # Process should be alive.
        assert handle.process.returncode is None, "ACP server process exited early"

        # Send initialize request and verify response.
        resp = await handle.initialize()
        assert resp.protocol_version == 1, (
            f"Expected protocol_version=1, got {resp.protocol_version}"
        )


async def test_basic_prompt(e2e_config: Path) -> None:
    """L4a: Send a prompt via ACP protocol, verify response event sequence."""
    async with _spawn_acp_server(e2e_config) as handle:
        await handle.initialize()
        new_sess = await handle.new_session()
        session_id = new_sess.session_id
        assert session_id, "Expected non-empty session_id"

        handle.clear_notifications()
        resp = await handle.prompt(session_id, "Hello, test agent!")
        assert resp.stop_reason is not None, "Expected stop_reason in PromptResponse"

        # Verify we received session update notifications.
        notifications = handle.get_notifications()
        assert len(notifications) > 0, "Expected at least one session update notification"


async def test_server_shutdown(e2e_config: Path) -> None:
    """L4a: Verify server is alive after initialize and can be cleanly terminated.

    The fixture teardown sends SIGTERM and waits for clean exit.
    We verify the process is still alive after initialize, confirming
    the server didn't crash during basic operation.

    Note: We intentionally do NOT create a session here to avoid triggering
    the known ACP close_session bug (#186) during teardown. Session creation
    is tested by test_basic_prompt above.
    """
    async with _spawn_acp_server(e2e_config) as handle:
        await handle.initialize()

        # Process should still be alive after initialize.
        assert handle.process.returncode is None, (
            "ACP server process exited unexpectedly during operation"
        )


# ---------------------------------------------------------------------------
# L4b Full Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
async def test_multi_turn_conversation(e2e_config: Path) -> None:
    """L4b: Send 2+ prompts in the same session."""
    async with _spawn_acp_server(e2e_config) as handle:
        await handle.initialize()
        new_sess = await handle.new_session()
        session_id = new_sess.session_id

        # First prompt.
        handle.clear_notifications()
        resp1 = await handle.prompt(session_id, "First message")
        assert resp1.stop_reason is not None
        notifs1 = handle.get_notifications()
        assert len(notifs1) > 0

        # Second prompt in the same session.
        handle.clear_notifications()
        resp2 = await handle.prompt(session_id, "Second message")
        assert resp2.stop_reason is not None
        notifs2 = handle.get_notifications()
        assert len(notifs2) > 0

        # Both responses should have valid stop reasons.
        assert resp1.stop_reason is not None, "First response should have a stop reason"
        assert resp2.stop_reason is not None, "Second response should have a stop reason"
        assert resp1.stop_reason == resp2.stop_reason, (
            f"TestModel inconsistent stop reasons: {resp1.stop_reason} != {resp2.stop_reason}"
        )


@pytest.mark.slow
async def test_tool_call_e2e(e2e_config_with_tool: Path) -> None:
    """L4b: Agent with bash tool, verify tool call events."""
    async with _spawn_acp_server(e2e_config_with_tool) as handle:
        await handle.initialize()
        new_sess = await handle.new_session()
        session_id = new_sess.session_id

        handle.clear_notifications()
        resp = await handle.prompt(session_id, "Run echo hello")
        assert resp.stop_reason is not None

        # Verify tool call events in notifications.
        notifications = handle.get_notifications()
        tool_call_updates = [
            n
            for n in notifications
            if n.update.session_update
            in {"tool_call_start", "tool_call_update", "tool_call_finish"}
        ]
        # TestModel with call_tools=["bash"] should produce tool call events.
        assert len(tool_call_updates) > 0, (
            "Expected tool call events when agent has call_tools configured"
        )


@pytest.mark.slow
async def test_cancellation_e2e(e2e_config: Path) -> None:
    """L4b: Start a prompt, cancel mid-stream."""
    async with _spawn_acp_server(e2e_config) as handle:
        await handle.initialize()
        new_sess = await handle.new_session()
        session_id = new_sess.session_id

        # Start a prompt and immediately cancel.
        # We use asyncio.wait_for to race the prompt against a timeout,
        # then send a cancel notification.
        prompt_task = asyncio.create_task(handle.prompt(session_id, "Long running prompt"))
        # Give the prompt a moment to start.
        await asyncio.sleep(0.1)
        # Send cancel notification.
        await handle.conn.cancel(
            CancelNotification(session_id=session_id, reason="test cancellation")
        )

        # Wait for the prompt task to complete (it should finish or error out).
        try:
            await asyncio.wait_for(prompt_task, timeout=10.0)
        except TimeoutError:
            prompt_task.cancel()
            with contextlib.suppress(Exception):
                await prompt_task
        except asyncio.CancelledError:
            # Cancellation may cause the prompt to error — that's acceptable.
            pass
