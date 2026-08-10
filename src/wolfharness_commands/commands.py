"""Slashed command to list nodes."""

from __future__ import annotations

from slashed import CommandContext  # noqa: TC002

from wolfharness.log import get_logger
from wolfharness.messaging.context import NodeContext  # noqa: TC001
from wolfharness_commands.base import NodeCommand
from wolfharness_commands.markdown_utils import format_table


logger = get_logger(__name__)


class ListNodesCommand(NodeCommand):
    """List all nodes in the pool with their status."""

    name = "list-nodes"
    category = "pool"

    async def execute_command(
        self,
        ctx: CommandContext[NodeContext],
        show_connections: bool = False,
    ) -> None:
        """List all nodes and their current status.

        Args:
            ctx: Command context with node
            show_connections: Whether to show node connections
        """
        node = ctx.get_data()
        assert node.pool

        rows = []
        for name, config in node.pool.manifest.agents.items():
            rows.append({
                "Node": name,
                "Status": "N/A",
                "Connections": "",
                "Description": config.description or "",
            })

        headers = ["Node", "Status", "Connections", "Description"]
        table = format_table(headers, rows)
        await ctx.print(f"## 🔗 Available Nodes\n\n{table}")
