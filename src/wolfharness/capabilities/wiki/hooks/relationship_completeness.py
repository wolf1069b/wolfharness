"""Confirmation-time relationship completeness validation."""

from __future__ import annotations

import re

from wolfharness.capabilities.wiki.section_constants import SECTION_CONTROLLER_IDENTITY

from .base import BaseHook, HookResult


_PLACEHOLDER_RE = re.compile(
    r"待\s*(?:URI|Fault|Procedure|Component|Symptom|DTC)?\s*(?:关联)?补充|待后续补充",
    re.IGNORECASE,
)


def _extract_frontmatter(content: str) -> str:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return ""
    for index, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            return "\n".join(lines[1:index])
    return ""


def _field_value(frontmatter: str, field: str) -> str:
    lines = frontmatter.splitlines()
    for index, line in enumerate(lines):
        match = re.match(rf"^{re.escape(field)}\s*:\s*(.*)$", line)
        if match is None:
            continue
        inline = match.group(1).strip()
        if inline and inline not in {"[]", "null", "~", "''", '""'}:
            return inline
        items: list[str] = []
        for following in lines[index + 1 :]:
            if following and not following[0].isspace():
                break
            stripped = following.strip()
            if stripped.startswith("- "):
                items.append(stripped[2:].strip())
        return "\n".join(items)
    return ""


class RelationshipCompletenessHook(BaseHook):
    """Block ``confirmed`` entities whose required graph edges are missing."""

    @property
    def name(self) -> str:
        return "relationship_completeness"

    def check(
        self,
        content: str,
        concept: str = "",
        class_name: str = "",
        object_name: str = "",
    ) -> HookResult:
        frontmatter = _extract_frontmatter(content)
        if _field_value(frontmatter, "status").strip("\"'") != "confirmed":
            return HookResult(
                hook_name=self.name,
                passed=True,
                message="Relationship completeness is enforced when status is confirmed.",
            )

        missing: list[str] = []
        if _field_value(frontmatter, "profile_id"):
            if not _field_value(frontmatter, "device_refs"):
                missing.append("device_refs")
            if not _field_value(frontmatter, "direct_component_uri"):
                missing.append("direct_component_uri")
            if not _field_value(frontmatter, "possible_faults"):
                missing.append("possible_faults")
        elif concept == "Device":
            if not _field_value(frontmatter, "critical_components"):
                missing.append("critical_components")
            if not _field_value(frontmatter, "symptom_refs"):
                missing.append("symptom_refs")
        elif concept == "DTC":
            controller_section = (
                content.split(f"## {SECTION_CONTROLLER_IDENTITY}", 1)[-1]
                if f"## {SECTION_CONTROLLER_IDENTITY}" in content
                else ""
            )
            if not _field_value(frontmatter, "controller_role"):
                missing.append("controller_role")
            if not _field_value(frontmatter, "controller_component") and not re.search(
                r"open_gap|未提供|未确认|未说明|未知",
                controller_section,
                re.IGNORECASE,
            ):
                missing.append("controller_component or explicit controller identity gap")
            if not _field_value(frontmatter, "related_faults"):
                missing.append("related_faults")
        elif concept == "Fault":
            if not _field_value(frontmatter, "affected_components"):
                missing.append("affected_components")
            if not (
                _field_value(frontmatter, "verification_procedures")
                or _field_value(frontmatter, "repair_procedures")
            ):
                missing.append("verification_procedures|repair_procedures")
        elif concept == "Procedure":
            if not _field_value(frontmatter, "target_components"):
                missing.append("target_components")
            if class_name == "diagnosis" and not _field_value(frontmatter, "procedure_scope"):
                missing.append("procedure_scope")

        if _PLACEHOLDER_RE.search(content):
            missing.append("unresolved_placeholders")

        if missing:
            return HookResult(
                hook_name=self.name,
                passed=False,
                message=(
                    f"Entity cannot become confirmed; missing or unresolved: {', '.join(missing)}."
                ),
                severity="error",
            )
        return HookResult(
            hook_name=self.name,
            passed=True,
            message="Confirmed entity has the required relationship fields.",
        )
