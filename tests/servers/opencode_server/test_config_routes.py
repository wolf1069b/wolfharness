"""Tests for /mode route (Phase 3)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from wolfharness.agents.modes import ModeCategory, ModeInfo
from wolfharness_server.opencode_server.routes.config_routes import list_modes


pytestmark = pytest.mark.integration


class TestListModes:
    """Test dynamic /mode route."""

    async def test_mode_returns_dynamic_modes(self):
        """/mode returns modes from agent.get_modes()."""
        agent = MagicMock()
        agent.get_modes = AsyncMock(
            return_value=[
                ModeCategory(
                    id="mode",
                    name="Mode",
                    available_modes=[
                        ModeInfo(id="default", name="Default"),
                        ModeInfo(id="accept_edits", name="Accept Edits"),
                    ],
                    current_mode_id="default",
                    category="mode",
                )
            ]
        )
        state = MagicMock()
        state.agent = agent

        result = await list_modes(state)  # type: ignore[arg-type]

        assert len(result) == 2
        names = {m.name for m in result}
        assert "default" in names
        assert "accept_edits" in names

    async def test_mode_fallback_on_error(self):
        """/mode returns default when get_modes() raises."""
        agent = MagicMock()
        agent.get_modes = AsyncMock(side_effect=RuntimeError("boom"))
        state = MagicMock()
        state.agent = agent

        result = await list_modes(state)  # type: ignore[arg-type]

        assert len(result) == 1
        assert result[0].name == "default"

    async def test_mode_empty_modes(self):
        """/mode returns default when no mode category found."""
        agent = MagicMock()
        agent.get_modes = AsyncMock(return_value=[])
        state = MagicMock()
        state.agent = agent

        result = await list_modes(state)  # type: ignore[arg-type]

        assert len(result) == 1
        assert result[0].name == "default"

    async def test_mode_no_mode_category(self):
        """/mode returns default when category has no mode id."""
        agent = MagicMock()
        agent.get_modes = AsyncMock(
            return_value=[
                ModeCategory(
                    id="model",
                    name="Model",
                    available_modes=[
                        ModeInfo(id="gpt-4o", name="GPT-4o"),
                    ],
                    current_mode_id="gpt-4o",
                    category="model",
                )
            ]
        )
        state = MagicMock()
        state.agent = agent

        result = await list_modes(state)  # type: ignore[arg-type]

        assert len(result) == 1
        assert result[0].name == "default"
