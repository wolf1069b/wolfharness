"""Confirmation-time diagnostic graph closure hook."""

from __future__ import annotations

from wolfharness.capabilities.wiki.quality import (
    confirmation_requirements,
    entity_status,
    has_unresolved_placeholder,
)

from .base import BaseHook, HookResult


class DiagnosticClosureHook(BaseHook):
    """Block confirmation while a diagnostic traversal edge is missing."""

    @property
    def name(self) -> str:
        return "diagnostic_closure"

    def check(
        self,
        content: str,
        concept: str = "",
        class_name: str = "",
        object_name: str = "",
    ) -> HookResult:
        if entity_status(content) != "confirmed":
            return HookResult(
                hook_name=self.name,
                passed=True,
                message="Diagnostic closure is enforced when status is confirmed.",
            )

        missing = [
            check.code
            for check in confirmation_requirements(content, concept, class_name)
            if not check.complete
        ]
        if has_unresolved_placeholder(content):
            missing.append("content.unresolved_placeholder")
        if missing:
            return HookResult(
                hook_name=self.name,
                passed=False,
                message=f"Diagnostic graph is incomplete: {', '.join(missing)}.",
                severity="error",
            )
        return HookResult(
            hook_name=self.name,
            passed=True,
            message="Diagnostic graph requirements are complete.",
        )
