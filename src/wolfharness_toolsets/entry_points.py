"""Entry point based toolset implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from epregistry import EntryPointRegistry

from wolfharness.capabilities.function_toolset import FunctionToolsetCapability
from wolfharness.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Sequence

    from wolfharness.tools.base import Tool


logger = get_logger(__name__)


class EntryPointTools(FunctionToolsetCapability):
    """Provider for entry point based tools."""

    def __init__(self, module: str) -> None:
        super().__init__(name=module)
        self.module = module
        self.registry = EntryPointRegistry[Callable[..., Any]]("wolfharness")

    async def get_tools(self) -> Sequence[Tool]:
        """Get tools from entry points."""
        # Return cached tools if available
        if self._tools:
            return self._tools

        self._tools = []
        entry_point = self.registry.get("tools")
        if not entry_point:
            msg = f"No tools entry point found for {self.module}"
            raise ValueError(msg)

        get_tools = entry_point.load()
        for item in get_tools():
            meta = {"module": self.module}
            tool = self.create_tool(item, metadata=meta)
            self._tools.append(tool)
        return self._tools
