"""Skill tool manager — dynamic Python tool import from SkillToolConfig."""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

from wolfharness.tools.base import Tool


if TYPE_CHECKING:
    from wolfharness_config.skills import SkillToolConfig

logger = logging.getLogger(__name__)


class SkillToolManager:
    """Manages dynamic import of Python tools from SkillToolConfig.

    Provides per-call import isolation — each import_tool call performs a fresh
    importlib.import_module, with no global caching. Failures are always handled
    gracefully via warning logs and None returns (never raises).
    """

    def import_tool(self, config: SkillToolConfig) -> Tool | None:
        """Import a single Python tool from a SkillToolConfig.

        Resolves the import path by splitting on the last colon (``:``),
        dynamically imports the module, and retrieves the named callable.

        Args:
            config: SkillToolConfig with type="python" and import_path.

        Returns:
            A Tool wrapping the imported callable, or None on failure.
        """
        import_path = config.import_path

        if ":" not in import_path:
            logger.warning(
                "Invalid import_path format (missing colon): %r. "
                "Expected 'module.path:function_name'.",
                import_path,
            )
            return None

        module_path, func_name = import_path.rsplit(":", 1)

        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            logger.warning(
                "Failed to import module %r for tool import_path=%r: %s",
                module_path,
                import_path,
                exc,
            )
            return None

        try:
            callable_obj = getattr(module, func_name)
        except AttributeError as exc:
            logger.warning(
                "Module %r has no attribute %r for tool import_path=%r: %s",
                module_path,
                func_name,
                import_path,
                exc,
            )
            return None

        if not callable(callable_obj):
            logger.warning(
                "Attribute %r in module %r is not callable for tool import_path=%r",
                func_name,
                module_path,
                import_path,
            )
            return None

        return Tool.from_callable(callable_obj)

    def import_tools(self, configs: list[SkillToolConfig]) -> list[Tool]:
        """Import multiple Python tools from a list of SkillToolConfig entries.

        Each config is imported independently. Failures are skipped with a
        warning — only successfully imported tools are returned.

        Args:
            configs: List of SkillToolConfig entries to import.

        Returns:
            List of successfully imported Tool instances.
        """
        tools: list[Tool] = []
        for config in configs:
            tool = self.import_tool(config)
            if tool is not None:
                tools.append(tool)
        return tools
