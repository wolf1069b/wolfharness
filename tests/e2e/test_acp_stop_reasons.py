"""L4 subprocess E2E tests for ACP stop reasons (B10).

ACP v1 defines the following valid ``StopReason`` values (see
``src/acp/schema/agent_responses.py:25-31``): ``end_turn``,
``max_tokens``, ``max_turn_requests``, ``refusal``, ``cancelled``.

B10.1 tests ``end_turn`` (the default path via TestModel).
B10.2-B10.5 are ``[skip]`` because the ACP server's
``event_converter.py`` hardcodes ``stop_reason`` based on event type
(``StreamCompleteEvent`` → ``"end_turn"``/``"cancelled"``,
``RunErrorEvent`` → ``"refusal"``), NOT the model's ``finish_reason``.
Custom stop reasons cannot be injected via TestModel or FunctionModel.
See issue #188.

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


# ---------------------------------------------------------------------------
# B10.1 — stop_reason="end_turn" (non-skip)
# ---------------------------------------------------------------------------


async def test_stop_reason_end_turn(e2e_config: Path) -> None:
    """B10.1: Verify ``stop_reason="end_turn"`` for normal TestModel completion.

    Sends a prompt to a TestModel-backed agent and verifies that the
    ``PromptResponse.stop_reason`` is ``"end_turn"`` (the default for normal
    completion via TestModel).

    ACP v1 valid stop reasons: ``end_turn``, ``max_tokens``,
    ``max_turn_requests``, ``refusal``, ``cancelled``.
    """
    async with _spawn_acp_server(e2e_config) as handle:
        await handle.initialize()
        new_sess = await handle.new_session()
        session_id = new_sess.session_id

        handle.clear_notifications()
        resp = await handle.prompt(session_id, "Hello, test agent!")

        assert resp.stop_reason is not None, "Expected stop_reason in PromptResponse"
        assert resp.stop_reason == "end_turn", (
            f"Expected stop_reason='end_turn', got '{resp.stop_reason}'"
        )


# ---------------------------------------------------------------------------
# B10.2 — stop_reason="max_tokens" (skip)
# ---------------------------------------------------------------------------


async def test_stop_reason_max_tokens(e2e_config: Path) -> None:
    """B10.2: Verify ``stop_reason="max_tokens"`` when max tokens reached.

    Test intent: Send prompt that triggers stop reason ``max_tokens``, verify
    ``PromptResponse.stop_reason="max_tokens"``. ACP v1 valid stop reasons:
    ``end_turn``, ``max_tokens``, ``max_turn_requests``, ``refusal``,
    ``cancelled``.
    """
    pass  # noqa: PIE790


# ---------------------------------------------------------------------------
# B10.3 — stop_reason="max_turn_requests" (skip)
# ---------------------------------------------------------------------------


async def test_stop_reason_max_turn_requests(e2e_config: Path) -> None:
    """B10.3: Verify ``stop_reason="max_turn_requests"`` when max turn requests reached.

    Test intent: Send prompt that triggers stop reason ``max_turn_requests``,
    verify ``PromptResponse.stop_reason="max_turn_requests"``. ACP v1 valid
    stop reasons: ``end_turn``, ``max_tokens``, ``max_turn_requests``,
    ``refusal``, ``cancelled``.
    """
    pass  # noqa: PIE790


# ---------------------------------------------------------------------------
# B10.4 — stop_reason="refusal" (skip)
# ---------------------------------------------------------------------------


async def test_stop_reason_refusal(e2e_config: Path) -> None:
    """B10.4: Verify ``stop_reason="refusal"`` when model refuses request.

    Test intent: Send prompt that triggers stop reason ``refusal``, verify
    ``PromptResponse.stop_reason="refusal"``. ACP v1 valid stop reasons:
    ``end_turn``, ``max_tokens``, ``max_turn_requests``, ``refusal``,
    ``cancelled``.
    """
    pass  # noqa: PIE790


# ---------------------------------------------------------------------------
# B10.5 — stop_reason="cancelled" (skip)
# ---------------------------------------------------------------------------


async def test_stop_reason_cancelled(e2e_config: Path) -> None:
    """B10.5: Verify ``stop_reason="cancelled"`` when turn is cancelled.

    Test intent: Send prompt that triggers stop reason ``cancelled``, verify
    ``PromptResponse.stop_reason="cancelled"``. ACP v1 valid stop reasons:
    ``end_turn``, ``max_tokens``, ``max_turn_requests``, ``refusal``,
    ``cancelled``.
    """
    pass  # noqa: PIE790
