"""Frontmatter schema compliance hook.

Validates that entity frontmatter only contains fields allowed by the
concept's schema definition (design_729.md §3). Non-schema fields are
rejected at write time with severity ``"error"``, blocking the write
so the worker must fix field names before the draft lands on disk.

Allowed field sets are loaded dynamically from
``xeno_adp_agentic/wiki/templates/default_schema.yaml`` via
:mod:`wolfharness.capabilities.wiki.schema_loader`, eliminating hardcoded drift.
"""

from __future__ import annotations

import re

from wolfharness.capabilities.wiki.schema_loader import (
    get_all_concept_frontmatter_fields,
    get_profile_frontmatter_fields,
)

from .base import BaseHook, HookResult


# ── Allowed frontmatter fields ───────────────────────────────────────────
# Loaded once from the YAML schema (cached via lru_cache in schema_loader).
# ``_COMMON_FIELDS`` and ``_OP_FIELDS`` are fallbacks for concepts not
# present in the YAML (OP is a synthetic concept for OPA/OPS records).

_ALLOWED_FIELDS: dict[str, frozenset[str]] = get_all_concept_frontmatter_fields()

_PROFILE_FIELDS: frozenset[str] = get_profile_frontmatter_fields()

# Fallback for concepts absent from the YAML (e.g. OP).
_COMMON_FIELDS: frozenset[str] = frozenset(
    {
        "id",
        "title",
        "description",
        "status",
        "sources",
        "aliases",
        "class_name",
        "object_name",
    },
)

_OP_FIELDS: frozenset[str] = frozenset(
    {
        "id",
        "title",
        "description",
        "status",
        "sources",
    },
)

# Internal pipeline markers injected by merge-conflict detection; not part
# of the YAML schema but must not block rewrites of conflict-marked pages.
_INTERNAL_PIPELINE_FIELDS: frozenset[str] = frozenset(
    {
        "conflict_pending",
        "conflict_refs",
    },
)

# Fields that are universally optional across entity concepts (not OP).
_SHARED_OPTIONAL_FIELDS: frozenset[str] = frozenset(
    {
        "applicable_models",
    },
)

# Frontmatter delimiter pattern
_FM_DELIMITER_RE = re.compile(r"^---\s*$", re.MULTILINE)


def _extract_frontmatter_fields(content: str) -> tuple[set[str], bool]:
    """Extract field names from YAML frontmatter.

    Returns ``(field_names, is_profile)`` where ``is_profile`` indicates
    whether this looks like a Symptom Profile file (has ``parent_symptom``
    or ``profile_id`` field).
    """
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return set(), False

    fields: set[str] = set()
    is_profile = False
    in_fm = True

    for line in lines[1:]:
        if line.strip() == "---":
            break
        if not in_fm:
            continue
        # Match top-level field: "key:" or "key: value"
        # Skip nested items (indented under lists/dicts)
        if line and not line[0].isspace() and not line.startswith("#"):
            m = re.match(r"^(\w[\w_]*)\s*:", line)
            if m:
                field = m.group(1)
                fields.add(field)
                if field in ("profile_id", "parent_symptom"):
                    is_profile = True

    return fields, is_profile


class FrontmatterSchemaHook(BaseHook):
    """Check that entity frontmatter only contains schema-allowed fields.

    Non-schema fields are rejected with severity ``"error"`` at write time,
    so the worker must fix the field name (e.g. ``profile:`` → ``profile_resource:``)
    before the draft is written.
    """

    @property
    def name(self) -> str:
        return "frontmatter_schema"

    def check(
        self,
        content: str,
        concept: str = "",
        class_name: str = "",
        object_name: str = "",
    ) -> HookResult:
        fields, is_profile = _extract_frontmatter_fields(content)

        if not fields:
            return HookResult(
                hook_name=self.name,
                passed=True,
                message="No frontmatter found; schema check skipped.",
            )

        # Profile files use a different field set.
        # Detect via concept parameter (authoritative) OR frontmatter fields
        # (fallback for files missing profile_id/parent_symptom).
        if concept == "SymptomProfile" or is_profile:
            allowed = _PROFILE_FIELDS | _COMMON_FIELDS | _INTERNAL_PIPELINE_FIELDS
            extra = fields - allowed
            if extra:
                return HookResult(
                    hook_name=self.name,
                    passed=False,
                    message=(
                        f"Profile frontmatter has {len(extra)} non-schema field(s): {', '.join(sorted(extra))}. Fix the field names to match the schema; write is blocked."
                    ),
                    severity="error",
                )
            return HookResult(
                hook_name=self.name,
                passed=True,
                message="Profile frontmatter fields are schema-compliant.",
            )

        # Regular entity: check against concept-specific allowed fields.
        # OP is not in the YAML schema; fall back to its explicit field set,
        # then to _COMMON_FIELDS for any other unknown concept.
        if concept == "OP":
            allowed = _OP_FIELDS
        else:
            allowed = _ALLOWED_FIELDS.get(concept, _COMMON_FIELDS) | _SHARED_OPTIONAL_FIELDS
        allowed = allowed | _INTERNAL_PIPELINE_FIELDS
        extra = fields - allowed

        if extra:
            return HookResult(
                hook_name=self.name,
                passed=False,
                message=(
                    f"Entity frontmatter has {len(extra)} non-schema "
                    f"field(s) for concept '{concept}': "
                    f"{', '.join(sorted(extra))}. "
                    f"Fix the field names to match the schema; write is blocked."
                ),
                severity="error",
            )

        return HookResult(
            hook_name=self.name,
            passed=True,
            message=f"Frontmatter fields are schema-compliant for '{concept}'.",
        )
