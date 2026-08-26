"""Shared validation harness used by capabilities and promotion gates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wolfharness.capabilities.wiki.hooks.body_sections import BodySectionsHook
from wolfharness.capabilities.wiki.hooks.component_taxonomy import ComponentTaxonomyHook
from wolfharness.capabilities.wiki.hooks.diagnostic_closure import DiagnosticClosureHook
from wolfharness.capabilities.wiki.hooks.directory_structure import DirectoryStructureHook
from wolfharness.capabilities.wiki.hooks.frontmatter_schema import FrontmatterSchemaHook
from wolfharness.capabilities.wiki.hooks.reference_integrity import ReferenceIntegrityHook
from wolfharness.capabilities.wiki.hooks.relationship_completeness import (
    RelationshipCompletenessHook,
)
from wolfharness.capabilities.wiki.hooks.uri_integrity import URIIntegrityHook


if TYPE_CHECKING:
    from collections.abc import Sequence

    from wolfharness.capabilities.wiki.hooks.base import BaseHook, HookResult


ENTITY_VALIDATION_HOOKS: tuple[BaseHook, ...] = (
    DirectoryStructureHook(),
    BodySectionsHook(),
    ComponentTaxonomyHook(),
    FrontmatterSchemaHook(),
    URIIntegrityHook(),
    ReferenceIntegrityHook(),
    RelationshipCompletenessHook(),
    DiagnosticClosureHook(),
)

# Hooks excluded from formal-write validation on confirmed/machine-validated
# content.  Both the MCP server and the capability layer use this set so
# the write gate is consistent across layers.
FORMAL_WRITE_EXCLUDED_HOOKS: frozenset[str] = frozenset(
    {"directory_structure", "body_sections", "diagnostic_closure"},
)


def run_entity_validation(
    *,
    content: str,
    concept: str,
    class_name: str,
    object_name: str,
    hooks: Sequence[BaseHook] = ENTITY_VALIDATION_HOOKS,
) -> list[HookResult]:
    """Run the deterministic validation chain."""
    return [
        hook.check(
            content=content,
            concept=concept,
            class_name=class_name,
            object_name=object_name,
        )
        for hook in hooks
    ]


def validation_feedback(results: Sequence[HookResult]) -> tuple[str, bool]:
    """Format failed hook results and indicate whether errors are present."""
    failures = [result for result in results if not result.passed]
    if not failures:
        return "", False

    errors = [result for result in failures if result.severity == "error"]
    warnings = [result for result in failures if result.severity != "error"]
    lines: list[str] = []
    if errors:
        lines.append(f"❌ {len(errors)} error(s) (write blocked):")
        lines.extend(
            f"  [{result.severity}] {result.hook_name}: {result.message}" for result in errors
        )
    if warnings:
        lines.append(f"⚠ {len(warnings)} warning(s):")
        lines.extend(
            f"  [{result.severity}] {result.hook_name}: {result.message}" for result in warnings
        )
    return "\n".join(lines), bool(errors)


def require_valid_entity(
    *,
    content: str,
    concept: str,
    class_name: str,
    object_name: str,
    hooks: Sequence[BaseHook] = ENTITY_VALIDATION_HOOKS,
) -> None:
    """Raise before a formal write when validation reports structural errors."""
    feedback, has_errors = validation_feedback(
        run_entity_validation(
            content=content,
            concept=concept,
            class_name=class_name,
            object_name=object_name,
            hooks=hooks,
        ),
    )
    if has_errors:
        raise ValueError(f"Entity validation failed before formal write:\n{feedback}")
