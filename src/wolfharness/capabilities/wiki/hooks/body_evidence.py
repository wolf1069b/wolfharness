"""Validate executable body sections and their raw evidence.

The wiki keeps standards and measurement facts in Markdown body sections,
not in mandatory frontmatter fields.  This hook therefore checks the body
contract that frontmatter-only relationship validation cannot see:

* composite diagnosis steps must link reusable Procedure entities;
* Profile mechanisms and diagnostic steps must use typed Fault/Procedure
  links or an explicit source-backed ``open_gap``;
* fact-bearing operation and criterion sections must retain a raw citation.

The hook only blocks confirmed entities.  Draft pages may be assembled in
dependency order and are checked again by the full audit before publication.
"""

from __future__ import annotations

import re

from wolfharness.capabilities.wiki.quality import (
    entity_status,
    extract_sections,
    extract_source_uris,
    extract_wiki_uris,
    has_usable_procedure_criteria,
    is_raw_source_uri,
    parse_frontmatter,
    wiki_uri_prefix,
)
from wolfharness.capabilities.wiki.section_constants import (
    GAP_RE,
    SECTION_COMMON_FAULTS,
    SECTION_DIAGNOSTIC_FLOW,
    SECTION_JUDGMENT_CRITERIA,
    SECTION_OPERATION_STEPS,
    SECTION_POSSIBLE_FAILURE,
)

from .base import BaseHook, HookResult


_LIST_ITEM_RE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.+?)\s*$")
_TOP_LEVEL_LIST_ITEM_RE = re.compile(r"^(?:\d+[.)]|[-*])\s+(.+?)\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*$")
_GAP_RE = re.compile(GAP_RE, re.IGNORECASE)


def _top_level_items(section: str) -> list[str]:
    """Return top-level Markdown list item bodies from a section."""
    items: list[str] = []
    for line in section.splitlines():
        if line[:1].isspace():
            continue
        match = _LIST_ITEM_RE.match(line)
        if match is not None:
            items.append(match.group(1).strip())
    return items


def _top_level_item_blocks(section: str) -> list[str]:
    """Return each top-level list item together with its continuation lines."""
    blocks: list[list[str]] = []
    for line in section.splitlines():
        match = _TOP_LEVEL_LIST_ITEM_RE.match(line)
        if match is not None:
            blocks.append([match.group(1).strip()])
        elif blocks and (not line.strip() or line[:1].isspace()):
            blocks[-1].append(line.strip())
    return ["\n".join(block).strip() for block in blocks]


def _has_raw_citation(section: str) -> bool:
    """Return whether a body section contains at least one raw URI."""
    return any(is_raw_source_uri(uri) for uri in extract_source_uris(section))


def _has_gap(item: str) -> bool:
    """Return whether an item explicitly records an unresolved source gap."""
    return _GAP_RE.search(item) is not None


def _missing_item_evidence(section: str) -> list[str]:
    """Find list items or specification rows without local raw evidence.

    ponytail: a section-level raw citation already backs the whole section;
    only flag items when the entire section lacks any raw URI or unresolved
    gap marker (paragraph-level granularity), not per-item URIs.
    """
    if _has_raw_citation(section) or _has_gap(section):
        return []
    missing = [item for item in _top_level_item_blocks(section) if not _has_gap(item)]
    table_rows = [line.strip() for line in section.splitlines() if "|" in line]
    if len(table_rows) > 1:
        data_rows = [row for row in table_rows[2:] if not _TABLE_SEPARATOR_RE.match(row)]
        missing.extend(row for row in data_rows if not _has_gap(row))
    return missing


def _section_source_has_raw(frontmatter: dict[str, object], section_name: str) -> bool:
    """Accept a structured section_sources citation as local evidence."""
    records = frontmatter.get("section_sources")
    if isinstance(records, dict):
        values = records.get(section_name, [])
        return any(is_raw_source_uri(uri) for uri in extract_source_uris(str(values)))
    if not isinstance(records, list):
        return False
    for record in records:
        if not isinstance(record, dict):
            continue
        section = record.get("section", record.get("target_section"))
        values = record.get("source_uri", record.get("source_uris", record.get("sources")))
        if section == section_name and any(
            is_raw_source_uri(uri) for uri in extract_source_uris(str(values))
        ):
            return True
    return False


def _parameter_rows_without_local_evidence(section: str) -> list[str]:
    """Find table rows whose parameter facts lack a row-local raw URI.

    ponytail: same section-level granularity as item evidence — a section
    with any raw citation (or an explicit gap marker) is considered sourced.
    """
    if _has_raw_citation(section) or _has_gap(section):
        return []
    table_rows = [line.strip() for line in section.splitlines() if "|" in line]
    if len(table_rows) < 3:
        return []
    data_rows = [row for row in table_rows[2:] if not _TABLE_SEPARATOR_RE.match(row)]
    return [row for row in data_rows if not _has_gap(row)]


class BodyEvidenceHook(BaseHook):
    """Check body-level diagnostic links and evidence citations."""

    @property
    def name(self) -> str:
        return "body_evidence"

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
                message="Body evidence closure is enforced when status is confirmed.",
            )

        sections = extract_sections(content)
        issues: list[str] = []
        frontmatter = parse_frontmatter(content)

        if concept == "Procedure":
            scope = frontmatter.get("procedure_scope")
            if class_name == "diagnosis" and scope == "composite":
                steps = _top_level_items(sections.get(SECTION_OPERATION_STEPS, ""))
                missing = [
                    step
                    for step in steps
                    if not _has_gap(step)
                    and not any(
                        uri.startswith(wiki_uri_prefix() + "/Procedure/")
                        for uri in extract_wiki_uris(step)
                    )
                ]
                if missing:
                    issues.append(
                        "Composite diagnosis steps without a Procedure URI or explicit open_gap: "
                        + "; ".join(missing[:3]),
                    )

            for section_name in (SECTION_OPERATION_STEPS, SECTION_JUDGMENT_CRITERIA):
                section = sections.get(section_name, "").strip()
                if not section:
                    issues.append(
                        f"Procedure section '{section_name}' is missing; add body content or an explicit open_gap.",
                    )
                    continue
                if _has_gap(section):
                    continue
                if section_name == SECTION_JUDGMENT_CRITERIA and not has_usable_procedure_criteria(
                    section
                ):
                    issues.append(
                        f"Procedure section '{SECTION_JUDGMENT_CRITERIA}' contains only a source pointer; write the actual pass/fail criterion in the body.",
                    )
                    continue
                missing_evidence = _missing_item_evidence(section)
                if missing_evidence and not _section_source_has_raw(frontmatter, section_name):
                    issues.append(
                        f"Procedure section '{section_name}' has steps/rows without local raw-source evidence: "
                        + "; ".join(missing_evidence[:3]),
                    )
                parameter_rows = _parameter_rows_without_local_evidence(section)
                if parameter_rows:
                    issues.append(
                        f"Procedure section '{section_name}' has parameter rows without a row-local raw-source locator: "
                        + "; ".join(parameter_rows[:3]),
                    )
                elif not _has_raw_citation(section) and not _section_source_has_raw(
                    frontmatter, section_name
                ):
                    issues.append(
                        f"Procedure section '{section_name}' contains executable facts but no raw-source citation.",
                    )

        elif concept == "Symptom" and "profile_id" in frontmatter:
            mechanisms = _top_level_items(sections.get(SECTION_POSSIBLE_FAILURE, ""))
            missing_faults = [
                item
                for item in mechanisms
                if not _has_gap(item)
                and not any(
                    uri.startswith(wiki_uri_prefix() + "/Fault/") for uri in extract_wiki_uris(item)
                )
            ]
            if missing_faults:
                issues.append(
                    "Profile mechanisms without a Fault URI or explicit open_gap: "
                    + "; ".join(missing_faults[:3]),
                )
            missing_mechanism_evidence = _missing_item_evidence(
                sections.get(SECTION_POSSIBLE_FAILURE, "")
            )
            if missing_mechanism_evidence and not _section_source_has_raw(
                frontmatter, SECTION_POSSIBLE_FAILURE
            ):
                issues.append(
                    "Profile mechanism items without local raw-source evidence: "
                    + "; ".join(missing_mechanism_evidence[:3]),
                )

            diagnostic_steps = _top_level_items(sections.get(SECTION_DIAGNOSTIC_FLOW, ""))
            missing_procedures = [
                item
                for item in diagnostic_steps
                if not _has_gap(item)
                and not any(
                    uri.startswith(wiki_uri_prefix() + "/Procedure/")
                    for uri in extract_wiki_uris(item)
                )
            ]
            if missing_procedures:
                issues.append(
                    "Profile diagnostic steps without a Procedure URI or explicit open_gap: "
                    + "; ".join(missing_procedures[:3]),
                )
            if (
                diagnostic_steps
                and not _has_raw_citation(sections.get(SECTION_DIAGNOSTIC_FLOW, ""))
                and not _section_source_has_raw(frontmatter, SECTION_DIAGNOSTIC_FLOW)
            ):
                issues.append(
                    f"Profile section '{SECTION_DIAGNOSTIC_FLOW}' has no raw-source citation."
                )
            missing_diagnostic_evidence = _missing_item_evidence(
                sections.get(SECTION_DIAGNOSTIC_FLOW, "")
            )
            if missing_diagnostic_evidence and not _section_source_has_raw(
                frontmatter, SECTION_DIAGNOSTIC_FLOW
            ):
                issues.append(
                    "Profile diagnostic items without local raw-source evidence: "
                    + "; ".join(missing_diagnostic_evidence[:3]),
                )

        elif concept == "Component":
            for section_name in ("性能参数", "specifications"):
                section = sections.get(section_name, "").strip()
                if not section or _has_gap(section):
                    continue
                if not _has_raw_citation(section):
                    issues.append(
                        f"Component section '{section_name}' contains facts but no raw-source citation.",
                    )
                parameter_rows = _parameter_rows_without_local_evidence(section)
                if parameter_rows:
                    issues.append(
                        f"Component section '{section_name}' has parameter rows without a row-local raw-source locator: "
                        + "; ".join(parameter_rows[:3]),
                    )

        elif concept == "Device":
            for section_name in (SECTION_COMMON_FAULTS, "控制器与故障码"):
                section = sections.get(section_name, "").strip()
                if not section or _has_gap(section):
                    continue
                missing_evidence = _missing_item_evidence(section)
                if missing_evidence and not _section_source_has_raw(frontmatter, section_name):
                    issues.append(
                        f"Device section '{section_name}' has rows without local raw-source evidence: "
                        + "; ".join(missing_evidence[:3]),
                    )
                elif not _has_raw_citation(section) and not _section_source_has_raw(
                    frontmatter,
                    section_name,
                ):
                    issues.append(
                        f"Device section '{section_name}' contains diagnostic links but no raw-source citation.",
                    )

        if issues:
            return HookResult(
                hook_name=self.name,
                passed=False,
                message=" | ".join(issues),
                severity="error",
            )
        return HookResult(
            hook_name=self.name,
            passed=True,
            message="Body diagnostic links and raw evidence are complete.",
        )
