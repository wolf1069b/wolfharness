"""Core command base classes.

This module provides the ``NodeCommand`` and ``AgentCommand`` base classes
that were previously in ``wolfharness_commands.base``.

They live in the core layer so that protocol servers can import them
without creating a server→cli dependency.
"""

from wolfharness.commands.base import AgentCommand, NodeCommand

__all__ = ["AgentCommand", "NodeCommand"]
