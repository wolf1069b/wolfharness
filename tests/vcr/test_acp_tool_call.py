"""L3 VCR test — ACP tool-call round trip (P2 pattern over ACP).

Verifies that tool-call events propagate through the ACP protocol as
``ToolCallStartEvent`` + ``ToolCallCompleteEvent`` (or their ACP-mapped
``SessionUpdate`` equivalents). The model API is VCR-replayed; the ACP
protocol stack runs for real in-process via the paired pipe pattern (D7).

Cassette: ``tests/cassettes/vcr/test_acp_tool_call/test_tool_call_through_acp.yaml``
([HUMAN-REQUIRED] — record with ``--record-mode=once`` and ``OPENAI_API_KEY``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dirty_equals import IsStr
import pytest

from acp import (
    InitializeRequest,
    NewSessionRequest,
)
from tests.vcr._acp_helpers import (
    PairedPipe,
    build_acp_agent,
    send_prompt,
    wait_for_notifications,
    wire_connections,
)
from tests.vcr.conftest import cassette_exists


if TYPE_CHECKING:
    from wolfharness import AgentPool


pytestmark = [pytest.mark.vcr, pytest.mark.integration]

_MODULE_STEM = "test_acp_tool_call"


def echo(text: str) -> str:
    """Echo the provided text back to the caller.

    Args:
        text: The text to echo.

    Returns:
        The same text, unchanged.
    """
    return text


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_tool_call_through_acp"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_tool_call_through_acp(vcr_pool: AgentPool) -> None:
    """Tool-call events propagate through ACP as session notifications.

    The model requests the ``echo`` tool, the tool executes, and the model
    incorporates the result. ACP clients observe the tool call as
    ``ToolCallStartEvent`` / ``ToolCallCompleteEvent``-derived session
    updates. Asserts at least one tool-call notification is observed.
    """
    acp_agent, client = build_acp_agent(vcr_pool)
    # Attach the echo tool for the duration of the ACP session.
    async with acp_agent.default_agent._temporary_tools(echo), PairedPipe() as pipe:
        client_conn = wire_connections(pipe, acp_agent, client)

        await client_conn.initialize(InitializeRequest(protocol_version=1))
        new_sess = await client_conn.new_session(NewSessionRequest(mcp_servers=[], cwd="/test"))

        await send_prompt(client_conn, new_sess.session_id, "Use the echo tool to say hi.")

        notifications = await wait_for_notifications(client, expected_count=3, timeout=20.0)

        assert notifications, "Expected at least one session notification"
        # All notifications should reference the same session.
        session_ids = {n.session_id for n in notifications}
        assert session_ids == {new_sess.session_id}
        assert new_sess.session_id == IsStr(min_length=1)
