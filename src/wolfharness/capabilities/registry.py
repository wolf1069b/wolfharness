"""Entry-point based capability discovery.

Provides ``discover_entry_point_capabilities()`` which loads all
capability classes registered via the ``wolfharness.capabilities`` entry-point
group. External packages can register new capability types by adding
an entry to their ``pyproject.toml``:

```toml
[project.entry-points."wolfharness.capabilities"]
my_cap = "my_package.my_module:MyCapability"
```

The factory calls ``discover_entry_point_capabilities()`` during
``compile()`` and stores the mapping for resolving YAML ``type:``
references to capability classes.

When a YAML config references a capability ``type:`` that is not in
the discovered registry, :class:`CapabilityNotFoundError` is raised
with an error message listing all available types.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pydantic_ai.capabilities import AbstractCapability


_ENTRY_POINT_GROUP = "wolfharness.capabilities"


class CapabilityNotFoundError(Exception):
    """Raised when a YAML ``type:`` references an unknown capability.

    The error message lists all available capability types discovered
    via entry points, helping users identify valid type values.

    Attributes:
        requested_type: The type name that was not found.
        available_types: Sorted list of all discovered type names.
    """

    def __init__(
        self,
        requested_type: str,
        available_types: list[str],
    ) -> None:
        """Initialize the error.

        Args:
            requested_type: The type name that was requested but not found.
            available_types: List of all available type names.
        """
        self.requested_type = requested_type
        self.available_types = sorted(available_types)
        types_str = ", ".join(self.available_types) if self.available_types else "(none)"
        msg = f"Unknown capability type: {requested_type!r}. Available types: {types_str}"
        super().__init__(msg)


def discover_entry_point_capabilities() -> dict[str, type[AbstractCapability[object]]]:
    """Discover capability classes registered via entry points.

    Loads all entry points in the ``wolfharness.capabilities`` group and
    returns a mapping from entry-point name to the loaded capability
    class. Each loaded value must be a subclass of
    :class:`~pydantic_ai.capabilities.AbstractCapability`.

    If duplicate names are found across packages, the first one
    discovered wins (entry points are iterated in registration order).

    Returns:
        Mapping from type name to capability class. Empty if no
        entry points are registered.
    """
    result: dict[str, type[AbstractCapability[object]]] = {}
    eps = entry_points(group=_ENTRY_POINT_GROUP)
    for ep in eps:
        if ep.name in result:
            continue
        loaded = ep.load()
        result[ep.name] = loaded
    return result


def resolve_capability_type(
    type_name: str,
    registry: dict[str, type[AbstractCapability[object]]] | None = None,
) -> type[AbstractCapability[object]]:
    """Resolve a type name to a capability class.

    Looks up ``type_name`` in the registry (or discovers entry points
    if no registry is provided). Raises
    :class:`CapabilityNotFoundError` if the type is not found.

    Args:
        type_name: The capability type name to resolve.
        registry: Optional pre-discovered registry. If ``None``,
            entry points are discovered fresh.

    Returns:
        The capability class corresponding to ``type_name``.

    Raises:
        CapabilityNotFoundError: If ``type_name`` is not in the registry.
    """
    resolved = registry if registry is not None else discover_entry_point_capabilities()
    if type_name not in resolved:
        raise CapabilityNotFoundError(type_name, list(resolved.keys()))
    return resolved[type_name]


__all__ = [
    "CapabilityNotFoundError",
    "discover_entry_point_capabilities",
    "resolve_capability_type",
]
