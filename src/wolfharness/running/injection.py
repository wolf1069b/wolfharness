"""Agent injection utilities."""

from __future__ import annotations

import inspect
import typing
from typing import TYPE_CHECKING, Any

from wolfharness.log import get_logger
from wolfharness.utils.inspection import get_fn_qualname


if TYPE_CHECKING:
    from collections.abc import Callable

    from wolfharness import AgentPool, MessageNode


logger = get_logger(__name__)


def is_node_type(typ: Any) -> bool:
    """Check if a type is or inherits from MessageNode."""
    from wolfharness import MessageNode

    if typ is MessageNode:
        return True

    # For "real" types
    if isinstance(typ, type):
        return issubclass(typ, MessageNode)

    # For generic types (Agent[T], etc)
    origin = getattr(typ, "__origin__", None)
    if origin is not None and isinstance(origin, type):
        return issubclass(origin, MessageNode)

    return False


class NodeInjectionError(Exception):
    """Raised when agent injection fails."""


def inject_nodes[T, **P](
    func: Callable[P, T],
    pool: AgentPool,
    provided_kwargs: dict[str, Any],
) -> dict[str, MessageNode[Any, Any]]:
    """Get nodes to inject based on function signature."""
    hints = typing.get_type_hints(func)
    params = inspect.signature(func).parameters
    logger.debug(
        "Injecting nodes", module=func.__module__, name=get_fn_qualname(func), type_hint=hints
    )

    nodes: dict[str, MessageNode[Any, Any]] = {}
    for name, param in params.items():
        if param.kind not in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            logger.debug("Skippin: wrong parameter kind", name=name, kind=param.kind)
            continue

        hint = hints.get(name)
        if hint is None:
            logger.debug("Skipping: no type hint", name=name)
            continue

        # Handle Optional/Union types
        origin = getattr(hint, "__origin__", None)
        args = getattr(hint, "__args__", ())

        # Check for MessageNode or any of its subclasses
        is_node = (
            is_node_type(hint)  # Direct node type
            or (  # Optional[Node[T]] or Union containing Node
                origin is not None and any(is_node_type(arg) for arg in args)
            )
        )

        if not is_node:
            msg = "Skipping. Not a node type."
            logger.debug(msg, name=name, hint=hint, origin=origin, args=args)
            continue

        logger.debug("Found node parameter", name=name)

        # Check for duplicate parameters
        if name in provided_kwargs and provided_kwargs[name] is not None:
            msg = (
                f"Cannot inject node {name!r}: Parameter already provided.\n"
                f"Remove the explicit argument or rename the parameter."
            )
            logger.error(msg)
            raise NodeInjectionError(msg)

        # Validate node name against manifest config
        if name not in pool.manifest.agents:
            available = ", ".join(sorted(pool.manifest.agents))
            msg = (
                f"No node named {name!r} found in configuration.\n"
                f"Available nodes: {available}\n"
                f"Check your YAML configuration or node name."
            )
            logger.error(msg)
            raise NodeInjectionError(msg)

        # Create the agent instance from config.
        # Pool-level agent storage was removed; we create instances on demand
        # from the manifest config via AnyAgentConfig.get_agent().
        config = pool.manifest.agents[name]
        node: MessageNode[Any, Any] = config.get_agent(pool=pool)
        nodes[name] = node
        logger.debug("Injected node from config", name=name)

    logger.debug("Injection complete.", nodes=sorted(nodes))
    return nodes
