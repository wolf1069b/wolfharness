"""Reference integrity validation hook.

Validates that wiki URI references in entity body content are well-formed
and flags placeholder text that indicates incomplete references.  Wiki
URIs use the configured wiki root (``<root_uri>/<Concept>/...``, e.g.
``viking://resources/<namespace>/...``).

Two types of checks:

1. **Placeholder detection** — flags body text like "待 URI 关联补充" or
   "待 Fault 关联补充" that indicates a relationship gap. These should be
   resolved by relation_worker when the target entity exists, not deleted by
   file_operator.

2. **URI format validation** — checks that wiki URIs in the body
   use 24-char hex hash format (not Chinese descriptive paths). This
   overlaps with URIIntegrityHook but focuses on body references rather
   than the entity heading.

3. **Dangling reference detection** — flags wiki URIs that appear
   in frontmatter relation fields but are empty lists or contain
   placeholder values.

Note: Actual existence checking (does the URI point to a real file?)
requires access to the WikiStore and is not performed by this hook.
That check should be done by the conductor in the post-processing phase.
"""

from __future__ import annotations

import re

from wolfharness.capabilities.wiki.quality import (
    extract_malformed_wiki_uris,
    is_wiki_uri,
    parse_frontmatter,
)

from .base import BaseHook, HookResult


# Body placeholder patterns indicating incomplete URI references
_PLACEHOLDER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"待\s*URI\s*关联补充"),
    re.compile(r"待\s*Fault\s*关联补充"),
    re.compile(r"待\s*Procedure\s*关联补充"),
    re.compile(r"待\s*Component\s*关联补充"),
    re.compile(r"待\s*Symptom\s*关联补充"),
    re.compile(r"待\s*DTC\s*关联补充"),
]

# Frontmatter relation fields that should contain URIs
_RELATION_FIELDS: frozenset[str] = frozenset(
    {
        "parent_symptom",
        "device_refs",
        "symptom_refs",
        "direct_component_uri",
        "critical_components",
        "affected_components",
        "verification_procedures",
        "repair_procedures",
        "assembly_parts",
        "target_components",
        "specification_refs",
        "related_faults",
        "controller_component",
        "possible_faults",
    },
)


def _uri_values(value: object) -> list[str]:
    """Collect scalar values from nested relation fields."""
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        values: list[str] = []
        for nested in value.values():
            values.extend(_uri_values(nested))
        return values
    if isinstance(value, list):
        values = []
        for nested in value:
            values.extend(_uri_values(nested))
        return values
    return []


def _extract_body(content: str) -> str:
    """Strip YAML frontmatter, return body only."""
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return content
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            return "\n".join(lines[i + 1 :])
    return content


def _extract_frontmatter(content: str) -> str:
    """Extract YAML frontmatter text."""
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return ""
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            return "\n".join(lines[1:i])
    return ""


class ReferenceIntegrityHook(BaseHook):
    """Check that URI references in entity content are complete and well-formed.

    Flags:
    - Placeholder text ("待 URI 关联补充" etc.) in body sections
    - Chinese descriptive paths in wiki URIs (should be hex hashes)
    - Empty relation fields in frontmatter that should contain URIs
    """

    @property
    def name(self) -> str:
        return "reference_integrity"

    def check(
        self,
        content: str,
        concept: str = "",
        class_name: str = "",
        object_name: str = "",
    ) -> HookResult:
        body = _extract_body(content)
        frontmatter = _extract_frontmatter(content)

        issues: list[str] = []
        invalid_relations: list[str] = []

        # ── 1. Placeholder detection in body ─────────────────────────────
        placeholders: list[str] = []
        for pattern in _PLACEHOLDER_PATTERNS:
            matches = pattern.findall(body)
            if matches:
                placeholders.extend(matches)

        if placeholders:
            issues.append(
                f"Found {len(placeholders)} placeholder reference(s) in body: "
                f"{', '.join(set(placeholders))[:200]}. "
                f"These are relationship gaps for relation_worker to resolve when evidence exists; "
                f"do not delete source-honest gap text just to silence the warning.",
            )

        # ── 2. URI format check in body ──────────────────────────────────
        bad_format_uris = [u for u in extract_malformed_wiki_uris(body) if u != ""]

        if bad_format_uris:
            issues.append(
                f"Found {len(bad_format_uris)} malformed wiki URI(s) with empty/object-less/placeholder tails: {', '.join(bad_format_uris[:3])}",
            )

        # ── 3. Empty relation fields in frontmatter ──────────────────────
        empty_relations: list[str] = []
        fm_lines = frontmatter.splitlines()
        for i, line in enumerate(fm_lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # Match "field: value" (inline value)
            m = re.match(r"^(\w[\w_]*)\s*:\s*(.*)", stripped)
            if m:
                field = m.group(1)
                value = m.group(2).strip()
                if field in _RELATION_FIELDS and value in ("[]", "''", '""', "", "~", "null"):
                    # Check if next line is a list item (YAML multiline list)
                    # If so, the field is NOT empty — it has items on following lines
                    has_list_items = False
                    for next_line in fm_lines[i + 1 :]:
                        next_stripped = next_line.strip()
                        if next_stripped == "---" or (next_line and not next_line[0].isspace()):
                            break
                        if next_stripped.startswith("- "):
                            has_list_items = True
                            break
                    if not has_list_items:
                        empty_relations.append(field)

        if empty_relations:
            issues.append(
                f"Empty relation field(s) in frontmatter: {', '.join(empty_relations)}. These should be filled by relation_worker.",
            )

        frontmatter_values = parse_frontmatter(content)
        for field in _RELATION_FIELDS:
            for value in _uri_values(frontmatter_values.get(field)):
                if is_wiki_uri(value):
                    continue
                invalid_relations.append(f"{field}={value}")
        if invalid_relations:
            issues.append(
                "Relation field(s) contain non-resolvable wiki identifiers: "
                + ", ".join(invalid_relations[:5])
                + ". Use a real wiki URI or leave an explicit open_gap.",
            )

        if issues:
            return HookResult(
                hook_name=self.name,
                passed=False,
                message=" | ".join(issues),
                severity="error" if invalid_relations else "warning",
            )

        return HookResult(
            hook_name=self.name,
            passed=True,
            message="All URI references are complete and well-formed.",
        )
