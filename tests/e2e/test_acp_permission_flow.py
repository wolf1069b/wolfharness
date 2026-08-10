"""L4 subprocess E2E tests for ACP permission flow (B6 group, all skip).

Tests session/request_permission flow:
    - B6.1 test_permission_approved [skip]
    - B6.2 test_permission_cancelled [skip]
    - B6.3 test_permission_cancel_during_pending [skip]

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


async def test_permission_approved(e2e_config: Path) -> None:
    """Test intent: Trigger a tool call that requires permission. Receive.

    ``session/request_permission`` notification with ``permission_id``,
    ``tool_name``, and ``params``. Send approval response with
    ``permission_id`` and ``decision="allow"``. Expect tool execution proceeds,
    ``tool_call_start`` and ``tool_call_end`` SessionUpdates emitted. Verify
    pending permission cleared after resolution.
    """


async def test_permission_cancelled(e2e_config: Path) -> None:
    """Test intent: Trigger a tool call requiring permission, receive.

    ``session/request_permission`` notification. Send cancellation response with
    ``permission_id`` and ``decision="deny"``. Expect tool execution skipped,
    no ``tool_call_end`` SessionUpdate for the cancelled tool. Verify prompt
    continues or returns with appropriate ``stop_reason``.
    """


async def test_permission_cancel_during_pending(e2e_config: Path) -> None:
    """Test intent: Trigger a tool call requiring permission, receive pending.

    ``session/request_permission``. Send ``session/cancel`` for the session
    while permission is pending. Expect session prompt returns with
    ``stop_reason="cancelled"`` and pending permission state cleared. Verify
    no tool execution occurs.
    """
