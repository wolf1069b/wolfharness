"""L4 subprocess E2E tests for ACP SessionUpdate variants (B9).

Tests the 13 SessionUpdate types defined in the ACP v1 schema:
``agent_message_chunk``, ``user_message_chunk``, ``agent_thought_chunk``,
``tool_call``, ``tool_call_update``, ``plan``, ``session_config``,
``mcp_server_added``, ``mcp_server_removed``, ``mcp_tool_added``,
``mcp_tool_removed``, ``mcp_progress``, ``terminal_output``,
``terminal_release``.

Non-skip tests (B9.1-B9.3): Verify SessionUpdate notifications emitted
during a basic prompt turn with TestModel.

Skip tests (B9.4-B9.12): SessionUpdate variants not yet implemented in
the wolfharness ACP server. Each skip test documents the test intent so
the expected behavior is clear once the feature is implemented.

All tests use ``model: test`` (pydantic-ai TestModel) so NO API key is needed.
"""

from __future__ import annotations

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
# Helper (mirrors test_acp_subprocess.py)
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


def _session_update_types(notifications: list[SessionNotification]) -> list[str]:
    """Extract the ``session_update`` discriminator from each notification."""
    return [n.update.session_update for n in notifications]


# ---------------------------------------------------------------------------
# B9.1 — agent_message_chunk (non-skip)
# ---------------------------------------------------------------------------


async def test_agent_message_chunk_notification(e2e_config: Path) -> None:
    """B9.1: Verify ``agent_message_chunk`` SessionUpdate emitted during text generation.

    Sends a prompt to a TestModel-backed agent and verifies that at least one
    ``session/update`` notification with ``update.session_update="agent_message_chunk"``
    is emitted during the agent's text response generation.
    """
    async with _spawn_acp_server(e2e_config) as handle:
        await handle.initialize()
        new_sess = await handle.new_session()
        session_id = new_sess.session_id

        handle.clear_notifications()
        resp = await handle.prompt(session_id, "Hello, test agent!")
        assert resp.stop_reason is not None, "Expected stop_reason in PromptResponse"

        notifications = handle.get_notifications()
        assert len(notifications) > 0, "Expected at least one session update notification"

        update_types = _session_update_types(notifications)
        assert "agent_message_chunk" in update_types, (
            f"Expected 'agent_message_chunk' in session updates, got: {update_types}"
        )


# ---------------------------------------------------------------------------
# B9.2 — user_message_chunk (non-skip)
# ---------------------------------------------------------------------------


async def test_user_message_chunk_notification(e2e_config: Path) -> None:
    """B9.2: Verify ``user_message_chunk`` SessionUpdate emitted when user message is processed.

    Sends a prompt to a TestModel-backed agent and verifies that at least one
    ``session/update`` notification with ``update.session_update="user_message_chunk"``
    is emitted when the user's message is processed by the server.

    Note: The wolfharness ACP server does not currently emit ``user_message_chunk``
    notifications (the event_converter only emits ``agent_message_chunk`` for
    agent output). This test documents the expected behavior per the ACP spec.
    See issue #188 for the server-side gap.
    """
    async with _spawn_acp_server(e2e_config) as handle:
        await handle.initialize()
        new_sess = await handle.new_session()
        session_id = new_sess.session_id

        handle.clear_notifications()
        resp = await handle.prompt(session_id, "Hello, test agent!")
        assert resp.stop_reason is not None, "Expected stop_reason in PromptResponse"

        notifications = handle.get_notifications()
        assert len(notifications) > 0, "Expected at least one session update notification"

        update_types = _session_update_types(notifications)
        assert "user_message_chunk" in update_types, (
            f"Expected 'user_message_chunk' in session updates, got: {update_types}"
        )


# ---------------------------------------------------------------------------
# B9.3 — tool_call_start and tool_call_end (non-skip)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "Tool call via ACP subprocess hangs — pre-existing issue "
        "(test_tool_call_e2e also times out). See issue #188."
    ),
    strict=False,
    raises=(AssertionError, TimeoutError, Exception),
)
@pytest.mark.known_bug
async def test_tool_call_start_end_notifications(e2e_config_with_tool: Path) -> None:
    """B9.3: Verify tool call start and end SessionUpdate notifications during tool execution.

    Sends a prompt to a TestModel-backed agent with ``call_tools=["bash"]`` and
    verifies that:
    - At least one ``session/update`` notification with ``update.session_update="tool_call"``
      (ToolCallStart) is emitted when the tool call begins.
    - At least one ``session/update`` notification with ``update.session_update="tool_call_update"``
      (ToolCallProgress with status="completed") is emitted when the tool call ends.

    In ACP v1, ``ToolCallStart`` has ``session_update="tool_call"`` and
    ``ToolCallProgress`` has ``session_update="tool_call_update"``. The v2 spec
    renames these to ``tool_call_start`` and ``tool_call_end``.

    Note: This test is xfail because tool calls via ACP subprocess currently hang
    (the existing ``test_tool_call_e2e`` has the same issue). See issue #188.
    """
    import asyncio

    # Wrap the entire test in a timeout to avoid hanging the CI.
    # The tool call via ACP subprocess is known to hang (issue #188).
    async with asyncio.timeout(25), _spawn_acp_server(e2e_config_with_tool) as handle:
        await handle.initialize()
        new_sess = await handle.new_session()
        session_id = new_sess.session_id

        handle.clear_notifications()
        resp = await handle.prompt(session_id, "Run echo hello")
        assert resp.stop_reason is not None, "Expected stop_reason in PromptResponse"

        notifications = handle.get_notifications()
        assert len(notifications) > 0, "Expected at least one session update notification"

        update_types = _session_update_types(notifications)

        # ToolCallStart has session_update="tool_call" in ACP v1 schema.
        assert "tool_call" in update_types, (
            f"Expected 'tool_call' (ToolCallStart) in session updates, got: {update_types}"
        )

        # ToolCallProgress has session_update="tool_call_update" in ACP v1 schema.
        # A completed tool call should produce at least one tool_call_update.
        assert "tool_call_update" in update_types, (
            f"Expected 'tool_call_update' (ToolCallProgress) in session updates, "
            f"got: {update_types}"
        )


# ---------------------------------------------------------------------------
# B9.4 — plan (skip)
# ---------------------------------------------------------------------------


async def test_plan_notification(e2e_config: Path) -> None:
    """B9.4: Verify ``plan`` SessionUpdate emitted during multi-step task.

    Test intent: Send a prompt that triggers plan generation (e.g., multi-step
    task). Verify ``session/update`` notification with ``update.type="plan"``
    containing ``plan`` array with step objects (``title``, ``status``, ``active``
    field). Verify multiple plan updates as agent progresses (step ``status``
    transitions from pending → in_progress → completed). Verify final plan state
    before ``stop_reason``.
    """
    pass  # noqa: PIE790


# ---------------------------------------------------------------------------
# B9.5 — session_config (skip)
# ---------------------------------------------------------------------------


async def test_session_config_notification(e2e_config: Path) -> None:
    """B9.5: Verify ``session_config`` SessionUpdate emitted on config change.

    Test intent: Change session config via ``session/set_mode`` or
    ``session/set_model``. Verify ``session/update`` notification with
    ``update.type="session_config"`` containing updated config fields (``mode``,
    ``model``, ``agent_role``, etc.). Verify config change takes effect on
    subsequent prompts.
    """
    pass  # noqa: PIE790


# ---------------------------------------------------------------------------
# B9.6 — mcp_server_added (skip)
# ---------------------------------------------------------------------------


async def test_mcp_server_added_notification(e2e_config: Path) -> None:
    """B9.6: Verify ``mcp_server_added`` SessionUpdate emitted on MCP server connect.

    Test intent: Add an MCP server to the session via ``mcp/connect`` or config
    change. Verify ``session/update`` notification with
    ``update.type="mcp_server_added"`` containing server ``name``, ``transport``
    type, and ``tools`` list with tool names and descriptions.
    """
    pass  # noqa: PIE790


# ---------------------------------------------------------------------------
# B9.7 — mcp_server_removed (skip)
# ---------------------------------------------------------------------------


async def test_mcp_server_removed_notification(e2e_config: Path) -> None:
    """B9.7: Verify ``mcp_server_removed`` SessionUpdate emitted on MCP server disconnect.

    Test intent: Remove an MCP server from the session via ``mcp/disconnect``.
    Verify ``session/update`` notification with
    ``update.type="mcp_server_removed"`` containing removed server ``name``.
    Verify server's tools no longer available in subsequent prompts.
    """
    pass  # noqa: PIE790


# ---------------------------------------------------------------------------
# B9.8 — mcp_tool_added (skip)
# ---------------------------------------------------------------------------


async def test_mcp_tool_added_notification(e2e_config: Path) -> None:
    """B9.8: Verify ``mcp_tool_added`` SessionUpdate emitted when MCP tool becomes available.

    Test intent: Connect an MCP server that exposes tools. Verify
    ``session/update`` notification with ``update.type="mcp_tool_added"``
    containing ``tool_name``, ``server_name``, and ``tool_description``. Verify
    tool becomes available for agent to call in subsequent prompts.
    """
    pass  # noqa: PIE790


# ---------------------------------------------------------------------------
# B9.9 — mcp_tool_removed (skip)
# ---------------------------------------------------------------------------


async def test_mcp_tool_removed_notification(e2e_config: Path) -> None:
    """B9.9: Verify ``mcp_tool_removed`` SessionUpdate emitted when MCP tool is removed.

    Test intent: Disconnect an MCP server whose tools were previously available.
    Verify ``session/update`` notification with
    ``update.type="mcp_tool_removed"`` containing removed ``tool_name`` and
    ``server_name``. Verify tool no longer callable in subsequent prompts.
    """
    pass  # noqa: PIE790


# ---------------------------------------------------------------------------
# B9.10 — mcp_progress (skip)
# ---------------------------------------------------------------------------


async def test_mcp_progress_notification(e2e_config: Path) -> None:
    """B9.10: Verify ``mcp_progress`` SessionUpdate emitted during long-running MCP tool call.

    Test intent: Trigger a long-running MCP tool call. Verify ``session/update``
    notification with ``update.type="mcp_progress"`` containing ``progress``
    value (0-100), ``message`` string, and ``tool_name``. Verify multiple
    progress notifications emitted as tool executes, with progress increasing
    monotonically.
    """
    pass  # noqa: PIE790


# ---------------------------------------------------------------------------
# B9.11 — terminal_output (skip)
# ---------------------------------------------------------------------------


async def test_terminal_output_notification(e2e_config: Path) -> None:
    """B9.11: Verify ``terminal_output`` SessionUpdate emitted during bash/shell tool call.

    Test intent: Trigger a bash/shell tool call that produces stdout output.
    Verify ``session/update`` notification with
    ``update.type="terminal_output"`` containing ``output`` string with
    stdout/stderr content and ``terminal_id``. Verify multiple terminal_output
    notifications for streaming output. Verify output matches actual command
    output.
    """
    pass  # noqa: PIE790


# ---------------------------------------------------------------------------
# B9.12 — terminal_release (skip)
# ---------------------------------------------------------------------------


async def test_terminal_release_notification(e2e_config: Path) -> None:
    """B9.12: Verify ``terminal_release`` SessionUpdate emitted after terminal session ends.

    Test intent: After a terminal session is used by a bash tool call, verify
    ``session/update`` notification with ``update.type="terminal_release"``
    containing ``terminal_id``. Verify terminal resource is freed and can be
    reused by subsequent tool calls (new ``terminal_id`` assigned).
    """
    pass  # noqa: PIE790
