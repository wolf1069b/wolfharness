"""Base command class with node-type filtering support.

Re-exports from ``wolfharness.commands.base`` for backward compatibility.
New code should import from ``wolfharness.commands.base`` directly.
"""

from __future__ import annotations

from wolfharness.commands.base import AgentCommand, NodeCommand


__all__ = ["AgentCommand", "NodeCommand"]
