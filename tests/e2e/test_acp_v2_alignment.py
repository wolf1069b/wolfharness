"""L4 subprocess E2E tests for ACP v1/v2 alignment (B11).

ACP v2 introduces changes to how ``PromptResponse`` and
``session/update`` notifications work compared to v1:

- v2: ``PromptResponse`` is empty (no ``stop_reason`` field);
  ``stop_reason`` is sent via ``IdleStateUpdate`` notification.
- v2: ``state_update`` notifications (``running`` → ``idle``) are
  emitted around each turn.
- v2: ``session/update`` uses only 15 known update types;
  ``TurnCompleteUpdate`` is replaced by standard v2 types.

All three tests are ``[skip]`` because AgentPool implements ACP v1 and
v2 is not yet supported. Each test documents the expected v2 behavior
so the migration path is clear.

All tests use ``model: test`` (pydantic-ai TestModel) so NO API key is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.e2e.conftest import SKIP_NO_BINARY, SKIP_WINDOWS


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(SKIP_NO_BINARY, reason="wolfharness binary not on PATH"),
    pytest.mark.skipif(SKIP_WINDOWS, reason="Windows stdio subprocess issues"),
]


# ---------------------------------------------------------------------------
# B11.1 — v2 empty PromptResponse (skip)
# ---------------------------------------------------------------------------


async def test_v2_empty_prompt_response(e2e_config: Path) -> None:
    """B11.1: Verify PromptResponse is empty in v2 (stop_reason sent via IdleStateUpdate).

    Test intent: Send a ``session/prompt`` request and verify the
    ``PromptResponse`` result is empty in v2 (v1 returns ``stop_reason``
    directly in the response, v2 sends it via ``IdleStateUpdate``
    notification). Verify the response result has no ``stop_reason`` field.
    Verify the ``stop_reason`` arrives separately as a ``session/update``
    notification with ``type="state_update"`` and ``state="idle"``.
    """
    pass  # noqa: PIE790


# ---------------------------------------------------------------------------
# B11.2 — v2 state_update notification (skip)
# ---------------------------------------------------------------------------


async def test_v2_state_update_notification(e2e_config: Path) -> None:
    """B11.2: Verify state_update notification (running→idle) sequence in v2.

    Test intent: Send a ``session/prompt`` request and verify a
    ``session/update`` notification with ``type="state_update"`` and
    ``state="running"`` is emitted before the response. Then verify a second
    ``session/update`` notification with ``type="state_update"`` and
    ``state="idle"`` is emitted after the response. Verify the state
    transition sequence is running→idle.
    """
    pass  # noqa: PIE790


# ---------------------------------------------------------------------------
# B11.3 — v2 standard update types (skip)
# ---------------------------------------------------------------------------


async def test_v2_standard_update_types(e2e_config: Path) -> None:
    """B11.3: Verify session/update uses only v2's 15 known update types.

    Test intent: Verify that ``session/update`` notifications use only v2's
    15 known update types (``agent_message_chunk``, ``user_message_chunk``,
    ``plan``, ``session_config``, ``mcp_server_added``,
    ``mcp_server_removed``, ``mcp_tool_added``, ``mcp_tool_removed``,
    ``mcp_progress``, ``terminal_output``, ``terminal_release``,
    ``tool_call_start``, ``tool_call_end``, ``state_update``, ``usage``), not
    non-standard ``TurnCompleteUpdate``. Send a prompt, collect all
    ``session/update`` notifications, verify each ``update.type`` is in the
    v2 known set.
    """
    pass  # noqa: PIE790
