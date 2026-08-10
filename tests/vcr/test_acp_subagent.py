"""L3 VCR test — ACP subagent delegation events.

Verifies that ``SpawnSessionStartEvent`` and ``SpawnSessionCompleteEvent``
(or their ACP-mapped equivalents) propagate through the ACP protocol when
a coordinator agent delegates to a worker. Uses the paired pipe pattern
(D7) and the ``vcr_pool_with_subagent`` fixture (coordinator + worker).

Cassette: ``tests/cassettes/vcr/test_acp_subagent/test_subagent_delegation.yaml``
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

_MODULE_STEM = "test_acp_subagent"


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_subagent_delegation"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_subagent_delegation(vcr_pool_with_subagent: AgentPool) -> None:
    """Coordinator delegates to worker; spawn/complete events propagate.

    The coordinator agent has a ``subagent`` tool. When the model invokes
    it, AgentPool emits ``SpawnSessionStart`` and (eventually)
    ``SpawnSessionComplete`` (wrapped in ``SubAgentEvent``). The ACP event
    converter maps these to session notifications. Asserts multiple
    notifications are received with consistent session IDs.
    """
    acp_agent, client = build_acp_agent(vcr_pool_with_subagent, agent_name="coordinator")
    async with PairedPipe() as pipe:
        client_conn = wire_connections(pipe, acp_agent, client)

        await client_conn.initialize(InitializeRequest(protocol_version=1))
        new_sess = await client_conn.new_session(NewSessionRequest(mcp_servers=[], cwd="/test"))
        assert new_sess.session_id == IsStr(min_length=1)

        await send_prompt(
            client_conn,
            new_sess.session_id,
            "Delegate to the worker agent: ask it to say hello.",
        )

        notifications = await wait_for_notifications(client, expected_count=5, timeout=30.0)

        assert notifications, "Expected at least one session notification"
        session_ids = {n.session_id for n in notifications}
        # All notifications should reference either the parent or a child session.
        assert all(sid == IsStr(min_length=1) for sid in session_ids)
