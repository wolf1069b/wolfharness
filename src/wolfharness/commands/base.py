"""Base command class with node-type filtering support."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

from slashed import SlashedCommand


if TYPE_CHECKING:
    from wolfharness.messaging import MessageNode


class NodeCommand(SlashedCommand, ABC):
    """Base class for commands with node-type filtering.

    Commands inheriting from this can override ``supports_node`` to restrict
    which node types they're available for.
    """

    name: str = ""

    @classmethod
    def supports_node(cls, node: MessageNode[Any, Any]) -> bool:
        """Check if this command supports the given node type."""
        return True


class AgentCommand(NodeCommand):
    """Base class for commands that only work with Agent nodes."""

    @classmethod
    def supports_node(cls, node: MessageNode[Any, Any]) -> bool:
        """Check if node is an Agent."""
        from wolfharness import Agent

        return isinstance(node, Agent)
