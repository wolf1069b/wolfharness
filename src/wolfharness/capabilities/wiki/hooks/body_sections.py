"""Body sections validation hook.

Validates that entity body contains all required sections per design_717.md
§3 Concept schemas, and that each section has non-empty content (not just
a heading with no body text).

Each Concept type has a defined set of required ``## `` sections. This hook
extracts all ``## `` headings from the body (excluding frontmatter), checks
that all required sections are present, and flags sections that are empty
(heading followed immediately by another heading or end of file with no
content between them).

For Symptom, the hook detects whether the content is a Canonical index.md
or a Profile file by checking for ``profile_id`` / ``parent_symptom`` in
frontmatter, and applies the appropriate section set.

Required section sets are loaded dynamically from
``xeno_adp_agentic/wiki/templates/default_schema.yaml`` via
:mod:`wolfharness.capabilities.wiki.schema_loader`, eliminating hardcoded drift.
``_CASE_REQUIRED_SECTIONS`` is a manual relaxation for fault-annotated /
case sources and is intentionally NOT in the YAML — it subtracts sections
those source types cannot provide.
"""

from __future__ import annotations

import re

import yaml

from wolfharness.capabilities.wiki.quality import is_case_source_uri
from wolfharness.capabilities.wiki.schema_loader import (
    get_all_concept_body_sections,
    get_profile_body_sections,
)
from wolfharness.capabilities.wiki.section_constants import (
    SECTION_DIAGNOSIS_FLOW,
    SECTION_JUDGMENT_CRITERIA,
    SECTION_MECHANISM,
    SECTION_OPERATION_STEPS,
    SECTION_OVERVIEW,
    SECTION_POSSIBLE_FAILURE,
    SECTION_SOURCE,
)

from .base import BaseHook, HookResult


# ── Required body sections ───────────────────────────────────────────────
# Loaded once from the YAML schema (cached via lru_cache in schema_loader).

_REQUIRED_SECTIONS: dict[str, frozenset[str]] = get_all_concept_body_sections()

_PROFILE_SECTIONS: frozenset[str] = get_profile_body_sections()

# Manual relaxation for fault-annotated/case sources — these sources lack
# teardown/assembly/manufacturer-spec sections.  Kept as explicit overrides
# because the YAML schema defines the *full* requirement; the case subset
# is a policy decision, not a schema property.
_CASE_REQUIRED_SECTIONS: dict[str, frozenset[str]] = {
    "Component": frozenset({SECTION_OVERVIEW, SECTION_MECHANISM, SECTION_SOURCE}),
    "DTC": frozenset({
        "故障码定义",
        SECTION_POSSIBLE_FAILURE,
        SECTION_DIAGNOSIS_FLOW,
        SECTION_SOURCE,
    }),
    "Procedure": frozenset({
        "操作目的",
        SECTION_OPERATION_STEPS,
        SECTION_JUDGMENT_CRITERIA,
        "后续处理",
        SECTION_SOURCE,
    }),
    "Device": frozenset({"基础信息", "包含系统", SECTION_SOURCE}),
}

# Section heading pattern: "## Title" (exactly 2 hashes, not 3+)
_SECTION_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)

# Frontmatter detection
_FM_START_RE = re.compile(r"^---\s*$", re.MULTILINE)


def _extract_body(content: str) -> str:
    """Strip YAML frontmatter, return body only."""
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return content
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            return "\n".join(lines[i + 1 :])
    return content  # No closing ---, return as-is


def _is_profile(content: str) -> bool:
    """Detect if content is a Symptom Profile file."""
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return False
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if "profile_id" in line or "parent_symptom" in line:
            return True
    return False


def _extract_sections(body: str) -> dict[str, str]:
    """Extract all ## level sections and their content.

    Returns ``{heading: content}`` where content is the text between
    this heading and the next ## heading (or end of body).
    """
    sections: dict[str, str] = {}
    matches = list(_SECTION_HEADING_RE.finditer(body))

    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = body[start:end].strip()
        sections[heading] = content

    return sections


class BodySectionsHook(BaseHook):
    """Check that entity body has all required sections with non-empty content.

    Per design_717.md §3, each Concept type has a defined set of required
    ``## `` sections. This hook:

    1. Detects whether the content is a regular entity or a Symptom Profile.
    2. Checks all required sections are present.
    3. Flags sections that are empty (heading with no body content).

    Sections are considered empty if they contain only whitespace, or only
    a single line like "待补充" / "待关联补充" without substantive content.
    This is a warning for required sections: keep the section and replace the
    line with a source-honest gap note when no evidence exists.
    """

    # Patterns that indicate placeholder/empty content.
    # "无" and "来源未说明" are valid source-honest gap notes, not placeholders.
    _PLACEHOLDER_PATTERNS = re.compile(
        r"^(待补充|待关联补充|待后续补充|—|-|\.\.\.)\s*$",
        re.MULTILINE,
    )

    # Component 工作机理 是身份核心字段：裸注记（来源未说明/无/—）不等于内容。
    # 真实来源缺失必须写成说明性 open_gap 并挂 gap OPA，见 _BARE_MECHANISM_GAP_RE。
    _BARE_MECHANISM_GAP_RE = re.compile(
        r"^(?:来源未说明|无|—|-|\.\.\.|n/?a)\s*[。.]?\s*$",
        re.IGNORECASE,
    )

    @property
    def name(self) -> str:
        return "body_sections"

    def check(
        self,
        content: str,
        concept: str = "",
        class_name: str = "",
        object_name: str = "",
    ) -> HookResult:
        body = _extract_body(content)

        # Determine section set: Profile vs regular entity
        is_profile = _is_profile(content)
        if is_profile:
            required = _PROFILE_SECTIONS
        else:
            frontmatter = _parse_frontmatter_for_source(content)
            source_values = frontmatter.get("sources", [])
            is_case_source = isinstance(source_values, list) and any(
                isinstance(source, str) and is_case_source_uri(source) for source in source_values
            )
            required = (
                _CASE_REQUIRED_SECTIONS.get(concept)
                if is_case_source
                else _REQUIRED_SECTIONS.get(concept)
            )

        if not required:
            return HookResult(
                hook_name=self.name,
                passed=True,
                message=f"No required sections defined for concept '{concept}'.",
            )

        sections = _extract_sections(body)

        # Check missing sections
        missing = required - set(sections.keys())
        # Check empty sections
        empty: list[str] = []
        mechanism_gap: list[str] = []
        for heading in required:
            if heading in sections:
                body_text = sections[heading]
                if not body_text or self._is_placeholder(body_text):
                    if concept == "Component" and heading == SECTION_MECHANISM:
                        mechanism_gap.append(heading)
                    else:
                        empty.append(heading)
                elif (
                    concept == "Component"
                    and heading == SECTION_MECHANISM
                    and self._is_bare_mechanism_gap(body_text)
                ):
                    mechanism_gap.append(heading)

        issues: list[str] = []
        if missing:
            issues.append(
                f"Missing {len(missing)} required section(s): {', '.join(sorted(missing))}",
            )
        if empty:
            issues.append(
                f"Empty or placeholder-only section(s): "
                f"{', '.join(sorted(empty))}. "
                f"Do not delete required sections; keep them and write a source-honest gap note "
                f"when the source does not provide the content.",
            )
        if mechanism_gap:
            message = (
                f"Component `{SECTION_MECHANISM}` 不得为空或裸占位"
                "（来源未说明/无/待补充）：按总成的功能/结构/常见动作推断，"
                "或从原文归纳；确实无从推断时写 open_gap 说明原因并挂 gap OPA。"
            )
            issues.append(message)

        if issues:
            severity = "error" if missing else "warning"
            return HookResult(
                hook_name=self.name,
                passed=False,
                message=" | ".join(issues),
                severity=severity,
            )

        return HookResult(
            hook_name=self.name,
            passed=True,
            message=(
                f"All {len(required)} required sections present and non-empty{' (Profile)' if is_profile else ''}."
            ),
        )

    def _is_placeholder(self, text: str) -> bool:
        """Check if section content is just a placeholder."""
        stripped = text.strip()
        if not stripped:
            return True
        if self._PLACEHOLDER_PATTERNS.match(stripped):
            return True
        return len(stripped) < 15 and ("待" in stripped and "补充" in stripped)

    def _is_bare_mechanism_gap(self, text: str) -> bool:
        """Whether ``工作机理`` is only a bare gap note (not a cited open_gap)."""
        return bool(self._BARE_MECHANISM_GAP_RE.match(text.strip()))


def _parse_frontmatter_for_source(content: str) -> dict[str, object]:
    """Read only the source field needed by the case-schema decision.

    This hook deliberately avoids importing the broader quality parser to
    keep validation hooks dependency-light and usable by the MCP boundary.
    """
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end = next((index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"), None)
    if end is None:
        return {}
    try:
        loaded = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError:
        return {}
    return loaded if isinstance(loaded, dict) else {}
