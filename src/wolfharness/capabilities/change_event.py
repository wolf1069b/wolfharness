"""ChangeEvent — emitted when a capability's resources change.

Part of the capability-native migration (M3). ``ChangeEvent`` is the
capability-layer equivalent of
:class:`~ChangeEvent`, using
the signal names from ``AbstractCapability`` (``tools_changed`` etc.)
as the ``kind`` discriminator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ChangeKind = Literal[
    "tools_changed",
    "prompts_changed",
    "resource_list_changed",
    "resource_updated",
    "skills_changed",
]
"""Discriminator for which resource type changed in a capability."""


@dataclass(frozen=True, slots=True)
class ChangeEvent:
    """Immutable event emitted when a capability's resources change.

    Attributes:
        capability_name: Name of the capability that emitted the event.
        kind: Which resource type changed. Widened to ``str`` to allow
            future event types without protocol changes. Use
            ``ChangeKind`` Literal for known values.
        source_uri: Optional URI of the source that changed, for
            URI-level routing.
    """

    capability_name: str
    kind: str = "tools_changed"
    source_uri: str = ""
