"""L4 subprocess E2E tests for ACP elicitation flow (B6 group, skip).

Tests elicitation/create notification:
    - B6.4 test_elicitation_create [skip]

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


async def test_elicitation_create(e2e_config: Path) -> None:
    """Test intent: Trigger a scenario where agent requests user input via.

    ``elicitation/create`` notification. Verify notification contains
    ``elicitation_id``, ``message``, and ``options`` array. Send elicitation
    response with ``elicitation_id`` and selected ``option``. Expect agent
    continues execution with provided input and produces final response.
    """
