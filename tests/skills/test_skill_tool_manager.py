"""Unit tests for SkillToolManager dynamic Python tool imports.

Tests cover import_tool() and import_tools() across these scenarios:
  1. Successful import → Tool returned
  2. Nonexistent module → warning logged, None returned
  3. Import path without colon → warning logged, None returned
  4. Non-callable attribute → warning logged, None returned
  5. Nonexistent attribute in valid module → warning logged, None returned
  6. Batch import: all valid → all returned
  7. Batch import: mixed valid/invalid → only valid returned
  8. Batch import: all invalid → empty list
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from wolfharness.skills.skill_tool_manager import SkillToolManager
from wolfharness.tools.base import Tool
from wolfharness_config.skills import SkillToolConfig


pytestmark = pytest.mark.unit


# ---- Fixtures ----


@pytest.fixture
def manager() -> SkillToolManager:
    """A fresh SkillToolManager per test."""
    return SkillToolManager()


# =========================================================================
# import_tool() scenarios
# =========================================================================


class TestImportTool:
    """Tests for SkillToolManager.import_tool()."""

    def test_import_json_loads(self, manager: SkillToolManager) -> None:
        """Importing json:loads returns a Tool wrapping json.loads."""
        config = SkillToolConfig(type="python", import_path="json:loads")
        tool = manager.import_tool(config)

        assert tool is not None
        assert isinstance(tool, Tool)
        assert tool.name == "loads"

    def test_import_os_getcwd(self, manager: SkillToolManager) -> None:
        """Importing os:getcwd returns a Tool wrapping os.getcwd."""
        config = SkillToolConfig(type="python", import_path="os:getcwd")
        tool = manager.import_tool(config)

        assert tool is not None
        assert isinstance(tool, Tool)
        assert tool.name == "getcwd"

    def test_import_nonexistent_module(self, manager: SkillToolManager) -> None:
        """Importing a nonexistent module logs a warning and returns None."""
        config = SkillToolConfig(
            type="python",
            import_path="nonexistent_module_xyz__test:nonexistent_func",
        )

        with patch("wolfharness.skills.skill_tool_manager.logger") as mock_logger:
            tool = manager.import_tool(config)

        assert tool is None
        mock_logger.warning.assert_called_once()

    def test_import_path_without_colon(self, manager: SkillToolManager) -> None:
        """Import path without ':' logs warning and returns None."""
        config = SkillToolConfig(type="python", import_path="os")

        with patch("wolfharness.skills.skill_tool_manager.logger") as mock_logger:
            tool = manager.import_tool(config)

        assert tool is None
        mock_logger.warning.assert_called_once()
        assert "missing colon" in mock_logger.warning.call_args[0][0].lower()

    def test_import_non_callable_attribute(self, manager: SkillToolManager) -> None:
        """Importing a non-callable attribute logs warning and returns None."""
        # os.sep is a plain string, not callable
        config = SkillToolConfig(type="python", import_path="os:sep")

        with patch("wolfharness.skills.skill_tool_manager.logger") as mock_logger:
            tool = manager.import_tool(config)

        assert tool is None
        mock_logger.warning.assert_called_once()
        assert "not callable" in mock_logger.warning.call_args[0][0].lower()

    def test_import_nonexistent_attribute(self, manager: SkillToolManager) -> None:
        """Importing a nonexistent attribute logs warning and returns None."""
        config = SkillToolConfig(
            type="python",
            import_path="os:this_attr_does_not_exist_xyz",
        )

        with patch("wolfharness.skills.skill_tool_manager.logger") as mock_logger:
            tool = manager.import_tool(config)

        assert tool is None
        mock_logger.warning.assert_called_once()
        assert "no attribute" in mock_logger.warning.call_args[0][0].lower()


# =========================================================================
# import_tools() batch scenarios
# =========================================================================


class TestImportTools:
    """Tests for SkillToolManager.import_tools()."""

    def test_batch_all_valid(self, manager: SkillToolManager) -> None:
        """Importing multiple valid configs returns all tools."""
        configs = [
            SkillToolConfig(type="python", import_path="json:loads"),
            SkillToolConfig(type="python", import_path="os:getcwd"),
        ]

        tools = manager.import_tools(configs)

        assert len(tools) == 2
        assert all(isinstance(t, Tool) for t in tools)
        assert {t.name for t in tools} == {"loads", "getcwd"}

    def test_batch_mixed_valid_invalid(self, manager: SkillToolManager) -> None:
        """Importing mixed valid/invalid configs returns only valid tools."""
        configs = [
            SkillToolConfig(type="python", import_path="json:loads"),
            SkillToolConfig(type="python", import_path="nonexistent_mod:func"),
            SkillToolConfig(type="python", import_path="os:getcwd"),
            SkillToolConfig(type="python", import_path="os:sep"),  # non-callable
        ]

        with patch("wolfharness.skills.skill_tool_manager.logger"):
            tools = manager.import_tools(configs)

        assert len(tools) == 2
        assert {t.name for t in tools} == {"loads", "getcwd"}

    def test_batch_all_invalid(self, manager: SkillToolManager) -> None:
        """Importing all invalid configs returns an empty list."""
        configs = [
            SkillToolConfig(type="python", import_path="nonexistent_mod:func"),
            SkillToolConfig(type="python", import_path="os:sep"),  # non-callable
        ]

        with patch("wolfharness.skills.skill_tool_manager.logger"):
            tools = manager.import_tools(configs)

        assert tools == []

    def test_batch_empty_configs(self, manager: SkillToolManager) -> None:
        """Importing an empty config list returns an empty list."""
        tools = manager.import_tools([])

        assert tools == []
