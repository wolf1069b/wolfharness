"""Base classes for post-tool-call wiki entity validation hooks.

Unlike the NMT harness hooks (which check translation pairs), wiki hooks
validate entity content after ``write_entity`` or ``patch_entity`` calls.
Each hook receives the entity's Markdown content and metadata, and returns
a :class:`HookResult` indicating pass/fail with a descriptive message.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HookResult:
    """Result of a wiki entity validation hook check.

    Attributes:
        hook_name: Name of the hook that produced this result.
        passed: Whether the check passed.
        message: Human-readable description of the result.
        severity: ``"error"`` or ``"warning"`` (default ``"warning"``).
    """

    hook_name: str
    passed: bool
    message: str
    severity: str = "warning"


class BaseHook(ABC):
    """Base class for post-tool-call wiki entity validation hooks.

    Each subclass implements :meth:`check` to validate entity content
    (Markdown with YAML frontmatter) and return a :class:`HookResult`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this hook."""
        ...

    @abstractmethod
    def check(
        self,
        content: str,
        concept: str = "",
        class_name: str = "",
        object_name: str = "",
    ) -> HookResult:
        """Check wiki entity content.

        Args:
            content: Full entity Markdown content (frontmatter + body).
            concept: Concept type (e.g. ``"Component"``, ``"Fault"``).
            class_name: Class name within the concept.
            object_name: Object name within the class.

        Returns:
            A :class:`HookResult` with pass/fail and a descriptive message.
        """
        ...
