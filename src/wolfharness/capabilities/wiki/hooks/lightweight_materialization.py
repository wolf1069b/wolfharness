"""Enforce lightweight, connected entity materialization.

The general wiki is a navigation and procedure layer, not a second copy of
model-specific parameter tables or wiring diagrams.  This hook runs for both
candidate and published writes so parameter-heavy pages are rejected before
they enter the store.
"""

from __future__ import annotations

import re

from wolfharness.capabilities.wiki.quality import (
    extract_sections,
    parse_frontmatter,
    wiki_uri_prefix,
)
from wolfharness.capabilities.wiki.section_constants import (
    SECTION_JUDGMENT_CRITERIA,
    SECTION_OPERATION_STEPS,
    SECTION_PREREQUISITES,
    SECTION_REQUIRED_TOOLS,
    SECTION_SOURCE,
)

from .base import BaseHook, HookResult


_MEASURED_VALUE_RE = re.compile(
    r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?\s*"
    r"(?:mV|kV|V|mA|A|Ω|ohm|MPa|kPa|bar|rpm|r/min|℃|°C|mm|cm|kg|"
    r"L/min|mL/min|N[·.]?m|Hz|ms|s|%)\b",
    re.IGNORECASE,
)
_WIRING_DETAIL_RE = re.compile(
    r"(?:针脚|针位|端子|Pin|线号|线色|导线).{0,24}"
    r"(?:\d+\s*(?:号|#)?|红|黑|白|蓝|黄|绿|棕|灰|紫|橙)",
    re.IGNORECASE,
)
_RAW_DELEGATION_RE = re.compile(
    r"(?:不在.{0,12}(?:复制|记录)|不(?:复制|记录)|以.{0,16}原始章节为准|"
    r"按.{0,16}原始章节|查阅.{0,16}原始章节)",
    re.IGNORECASE,
)


def _as_uri_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item.strip()]
    if isinstance(value, dict):
        return [item for item in value.values() if isinstance(item, str) and item.strip()]
    return []


def _has_concept_uri(values: list[str], concept: str) -> bool:
    prefix = f"{wiki_uri_prefix()}/{concept}/"
    return any(value.startswith(prefix) for value in values)


def _unstable_detail_lines(text: str) -> list[str]:
    """Return copied model-specific values or wiring-map details."""
    offending: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or _RAW_DELEGATION_RE.search(line):
            continue
        if _MEASURED_VALUE_RE.search(line) or _WIRING_DETAIL_RE.search(line):
            offending.append(line)
    return offending


class LightweightMaterializationHook(BaseHook):
    """Block parameter-heavy Procedure nodes and general pages carrying
    model-specific measured values or wiring maps.
    """

    @property
    def name(self) -> str:
        return "lightweight_materialization"

    def check(
        self,
        content: str,
        concept: str = "",
        class_name: str = "",
        object_name: str = "",
    ) -> HookResult:
        frontmatter = parse_frontmatter(content)
        sections = extract_sections(content)
        issues: list[str] = []

        if concept == "Procedure":
            target_components = _as_uri_list(frontmatter.get("target_components"))
            if not _has_concept_uri(target_components, "Component"):
                issues.append("Procedure must reference at least one real target Component URI")

            steps = sections.get(SECTION_OPERATION_STEPS, "").strip()
            if not steps or not re.search(r"^\s*(?:\d+[.)]|[-*])\s+\S", steps, re.MULTILINE):
                issues.append("Procedure must contain reusable ordered actions")

            checked_text = "\n".join(
                sections.get(name, "")
                for name in (
                    SECTION_PREREQUISITES,
                    SECTION_REQUIRED_TOOLS,
                    SECTION_OPERATION_STEPS,
                    SECTION_JUDGMENT_CRITERIA,
                )
            )
            unstable = _unstable_detail_lines(checked_text)
            if unstable:
                issues.append(
                    "Procedure contains unstable numeric/wiring detail; keep the reusable action "
                    "and delegate model-specific values, wire numbers, colors and pin maps to the "
                    "cited raw chapter: " + "; ".join(unstable[:3]),
                )

        elif concept in {"Component", "Device", "DTC", "Symptom"}:
            # General pages may retain identity codes and DTC codes, but not
            # model-specific measured values or wiring maps copied into body
            # sections.
            body_without_sources = "\n".join(
                body for name, body in sections.items() if name != SECTION_SOURCE
            )
            unstable = _unstable_detail_lines(body_without_sources)
            if unstable:
                issues.append(
                    f"{concept} contains unstable numeric/wiring detail; retain the general fact "
                    "and raw source link instead: " + "; ".join(unstable[:3]),
                )

        if issues:
            return HookResult(
                hook_name=self.name,
                passed=False,
                message=" | ".join(issues),
                severity="error",
            )

        # Keep this hook focused on materialization policy; graph closure for
        # other concepts remains in relationship/audit checks.
        return HookResult(
            hook_name=self.name,
            passed=True,
            message="Entity satisfies lightweight materialization policy.",
        )
