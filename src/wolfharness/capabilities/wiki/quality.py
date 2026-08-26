"""Deterministic diagnostic-closure checks for wiki entities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import os
import re
from typing import Literal, NotRequired, TypedDict, get_args

import yaml

from wolfharness.capabilities.wiki.namespaces import resources_root
from wolfharness.capabilities.wiki.schema_loader import get_concept_schema
from wolfharness.capabilities.wiki.section_constants import (
    PLACEHOLDER_TEXT_RE,
    SECTION_FAILURE_MECHANISM,
    SECTION_IMPACT_SCOPE,
    SECTION_MECHANISM,
    SECTION_POSSIBLE_FAILURE,
    SECTION_REPAIR_METHOD,
    SECTION_VERIFICATION,
)


_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_WIKI_CONCEPTS = frozenset({"Device", "Component", "DTC", "Symptom", "Fault", "Procedure", "OP"})


def _configured_root(env_name: str) -> str:
    """Return an env-configured root or a namespace that can never be persisted."""
    namespace = os.environ.get(env_name, "").strip()
    return resources_root(namespace) if namespace else "viking://unconfigured"


# Store creation replaces both roots before any entity validation. Keeping the
# pre-store boundary namespace-agnostic avoids embedding a deployment target in
# library code while still allowing modules to import before runtime config loads.
_wiki_root_uri = _configured_root("VIKING_NAMESPACE")
_raw_source_root_uri = _configured_root("VIKING_RAW_NAMESPACE")

# (concept or "_Profile") → {frontmatter_field: (body_section_heading, target_concept)}
_BODY_LINK_MAP: dict[str, dict[str, tuple[str, str]]] = {
    "Fault": {
        "affected_components": (SECTION_IMPACT_SCOPE, "Component"),
        "verification_procedures": (SECTION_VERIFICATION, "Procedure"),
        "repair_procedures": (SECTION_REPAIR_METHOD, "Procedure"),
    },
    "DTC": {
        "related_faults": (SECTION_POSSIBLE_FAILURE, "Fault"),
    },
    "_Profile": {
        "possible_faults": (SECTION_POSSIBLE_FAILURE, "Fault"),
    },
    "Procedure": {
        "target_components": ("操作目的", "Component"),
    },
    "Device": {
        "critical_components": ("关重件清单", "Component"),
    },
}


def _is_profile(content: str) -> bool:
    """Detect Symptom Profile by presence of ``profile_id`` in frontmatter."""
    fm = parse_frontmatter(content)
    return "profile_id" in fm


_WIKI_URI_RE: re.Pattern[str]
_MALFORMED_WIKI_URI_RE: re.Pattern[str]


def _compile_wiki_regexes(root_uri: str) -> None:
    """Recompile URI-matching regexes against a root URI prefix."""
    global _WIKI_URI_RE, _MALFORMED_WIKI_URI_RE  # noqa: PLW0603 - module singletons reconfigured at runtime
    esc = re.escape(root_uri.rstrip("/"))
    _WIKI_URI_RE = re.compile(
        rf"{esc}/(?:Device|Component|DTC|Symptom|Fault|Procedure|OP)"
        r"/(?:[0-9a-f]{24}|[^/#\s<>]+(?:/[^/#\s<>]+)*)"
        r"(?:#[^\s<>]+)?",
    )
    _MALFORMED_WIKI_URI_RE = re.compile(
        rf"{esc}/(?:Device|Component|DTC|Symptom|Fault|Procedure|OP)/"
        r"(?![0-9a-f]{24}(?:#|$))"
        r"([^/#\s\)\]\"'\u3000\u3001\u3002]*)"
        r"(?=$|[<{\)\]\s,;:!?。，；：！？])",
    )


def set_wiki_root_uri(uri: str) -> None:
    """Set the wiki root URI prefix used by all URI checks in this module."""
    global _wiki_root_uri  # noqa: PLW0603 - module config setter
    _wiki_root_uri = uri.rstrip("/")
    _compile_wiki_regexes(_wiki_root_uri)


def is_wiki_uri(uri: str) -> bool:
    return uri.startswith(_wiki_root_uri + "/")


def wiki_uri_prefix() -> str:
    return _wiki_root_uri


def set_raw_source_root_uri(uri: str) -> None:
    """Set the active raw-source root used by write-time validators."""
    global _raw_source_root_uri  # noqa: PLW0603 - module config setter
    _raw_source_root_uri = uri.rstrip("/")


# Initial regexes for the default prefix; recompiled by set_wiki_root_uri()
# when a store is created (viking backend → viking://resources/<namespace>).
_compile_wiki_regexes(_wiki_root_uri)
# Any RFC-3986 scheme: viking://, file://, kb://, http://, etc.
# Agents fetch source content from arbitrary MCP providers; the scheme is
# an opaque provenance identifier, not a hardcoded file-backend selector.
_URI_START_RE = re.compile(r"[a-z][a-z0-9+.\-]*://")
_URI_TRAILING_PUNCTUATION = ".,;:!?，。；：！？、"

# Raw manual chapter URIs.  Addressable backend paths
# ``viking://.../chapters/<subdir>/chapter.md``, which may contain CJK
# characters and spaces (``01_1 前言``).
_RAW_CHAPTER_URI_RE = re.compile(
    r"viking://[^\n)\]>]+?/chapters/[^\n)\]>]*?chapter\.md",
)

_RAW_CHAPTER_PREFIXES: tuple[str, ...] = ()
_RAW_CHAPTER_URIS: frozenset[str] = frozenset()

_CROSS_NAMESPACE_RAW_URI_RE = re.compile(
    r"^viking://resources/[^/]+/raw/(?P<relative>.+\.md)$",
)

_CROSS_NAMESPACE_BOM_URI_RE = re.compile(
    r"^viking://resources/[^/]+/bom/(?P<relative>.+\.md)$",
)

_EXTERNAL_URI_RE = re.compile(r"^(?!viking://|file://)[a-z][a-z0-9+.\-]*://")

_SOURCE_URI_PREFIX_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://")


def is_source_uri_scheme(value: str) -> bool:
    """Fast check for any ``scheme://`` URI prefix (viking, file, kb, http, …)."""
    return bool(_SOURCE_URI_PREFIX_RE.match(value))


BuildProfile = Literal["manual", "case"]
BUILD_PROFILES = frozenset(get_args(BuildProfile))


class RawSourceKind(StrEnum):
    """Supported provenance resource families."""

    MANUAL_CHAPTER = "manual_chapter"
    CASE = "case"
    EXTERNAL = "external"


class SourceReadStatus(StrEnum):
    """Result of resolving a provenance URI through its owning backend."""

    OK = "ok"
    NOT_FOUND = "not_found"
    UNAVAILABLE = "unavailable"
    INVALID_URI = "invalid_uri"


class IssueDisposition(StrEnum):
    """Workflow required to handle one deterministic audit finding."""

    REPAIR_ONLY = "repair_only"
    GAP = "gap"
    CONFLICT = "conflict"


@dataclass(frozen=True, slots=True)
class SourceReadResult:
    """Typed result for a raw-source read and content fingerprint."""

    uri: str
    kind: RawSourceKind | None
    status: SourceReadStatus
    content: str | None = None
    content_hash: str | None = None
    error_code: str = ""


def classify_raw_source_uri(
    uri: str,
    *,
    raw_root_uri: str | None = None,
) -> RawSourceKind | None:
    """Classify a provenance URI without binding it to one namespace.

    The active raw root is supplied by the caller, so local and remote builds
    share the same contract.  A chapter leaf is a manual source; any other
    Markdown leaf under the active raw root is a single-file case source.
    Directory URIs accept the shape of their conventional ``chapter.md``
    leaf.  Cross-namespace raw libraries are recognized by their public
    resource shape rather than a configured namespace or collection
    identifier.
    """
    if not isinstance(uri, str):
        return None
    value = uri.strip()
    if not value:
        return None
    # Workers may pass directory URIs (no ``.md`` leaf) or fragment-anchored
    # chapter URIs; normalize both before matching.
    value = value.partition("#")[0].rstrip("/")
    if value in _RAW_CHAPTER_URIS:
        return RawSourceKind.MANUAL_CHAPTER
    # A directory URI maps to its conventional leaf, mirroring the
    # ``chapters/<subdir>/chapter.md`` layout without a backend read.
    leaf = value if value.endswith(".md") else f"{value}/chapter.md"
    cross_namespace = _CROSS_NAMESPACE_RAW_URI_RE.fullmatch(leaf)
    if cross_namespace is not None:
        relative = cross_namespace.group("relative")
        if "/chapters/" in f"/{relative}" and relative.endswith("/chapter.md"):
            return RawSourceKind.MANUAL_CHAPTER
        return RawSourceKind.CASE
    if _CROSS_NAMESPACE_BOM_URI_RE.fullmatch(leaf) is not None:
        return RawSourceKind.CASE
    root = (raw_root_uri if raw_root_uri is not None else _raw_source_root_uri).rstrip("/")
    if not root or not value.startswith(root + "/"):
        if _EXTERNAL_URI_RE.match(value):
            return RawSourceKind.EXTERNAL
        return None
    relative = value.removeprefix(root + "/")
    if not relative:
        if _EXTERNAL_URI_RE.match(value):
            return RawSourceKind.EXTERNAL
        return None
    if "/chapters/" in f"/{relative}" and relative.endswith("/chapter.md"):
        return RawSourceKind.MANUAL_CHAPTER
    # A non-``.md`` path under the raw root is a directory source; classify by
    # shape — the caller verifies existence downstream.
    return RawSourceKind.CASE


def is_raw_chapter_uri(uri: str) -> bool:
    """Return ``True`` for any raw manual chapter URI (real backend path).

    Real backend paths match by their ``/chapters/…/chapter.md`` shape, which
    is distinct from wiki entity URIs (``…/Device/…``) and OPA resource URIs
    (``…/OP/…``).
    """
    if not isinstance(uri, str):
        return False
    return (
        uri in _RAW_CHAPTER_URIS
        or uri.startswith(_RAW_CHAPTER_PREFIXES)
        or ("/chapters/" in uri and uri.endswith("chapter.md"))
    )


def is_external_source_uri(uri: str) -> bool:
    """Return ``True`` for an external MCP source URI (non-viking/non-file scheme)."""
    return classify_raw_source_uri(uri) is RawSourceKind.EXTERNAL


_BOM_SOURCE_URI_RE = re.compile(r"^viking://resources/[^/]+/bom/(?P<relative>.+\.md)$")


def is_bom_source_uri(uri: str) -> bool:
    """Return ``True`` for a cross-namespace BOM library URI.

    The global BOM library (e.g. ``viking://resources/730/bom/component/…``) is
    a name-shape sibling of the raw chapter library (``…/raw/…``) but lives
    under ``/bom/``, so it is neither a raw-chapter URI nor an external one.
    Write-time validators must accept it as real provenance evidence for
    BOM-driven Component identity.
    """
    if not isinstance(uri, str):
        return False
    return _BOM_SOURCE_URI_RE.fullmatch(uri.strip()) is not None


def register_raw_chapter_uris(uris: list[str]) -> None:
    """Register exact manual leaves discovered from the active raw backend.

    Parsed PDF libraries may store chapter Markdown directly in a nested TOC
    tree instead of the legacy ``chapters/.../chapter.md`` layout. Exact URI
    registration keeps manual/case classification evidence-based without
    encoding tenant, model, or directory names in validation logic.
    """
    global _RAW_CHAPTER_URIS  # noqa: PLW0603 - process-wide validation registry
    normalized = frozenset(uri.strip() for uri in uris if uri.strip())
    _RAW_CHAPTER_URIS = _RAW_CHAPTER_URIS.union(normalized)


def is_case_source_uri(uri: str, *, raw_root_uri: str | None = None) -> bool:
    """Return whether *uri* identifies a supported single-file case source."""
    return classify_raw_source_uri(uri, raw_root_uri=raw_root_uri) is RawSourceKind.CASE


def is_raw_source_uri(uri: str, *, raw_root_uri: str | None = None) -> bool:
    """Return whether *uri* belongs to any supported provenance family."""
    return classify_raw_source_uri(uri, raw_root_uri=raw_root_uri) is not None


_UNRESOLVED_PLACEHOLDER_RE = re.compile(
    r"(?:"
    r"\b(?:TODO|TBD|FIXME|XXX)\b"
    r"|\{\{[^}]+\}\}"
    r"|<待[^>]*>"
    r"|<TODO[^>]*>"
    r")",
    re.IGNORECASE,
)


class QualityIssue(TypedDict):
    """One actionable corpus quality issue."""

    uri: str
    concept: str
    code: str
    severity: str
    message: str
    disposition: NotRequired[str]
    opa_reason_code: NotRequired[str]
    target_uri: NotRequired[str]
    gap_category: NotRequired[str]
    repair_action: NotRequired[str]


class CoverageSummary(TypedDict):
    """Completion counters for one relationship rule."""

    eligible: int
    complete: int
    percent: float


class WikiAuditReport(TypedDict):
    """Serializable result returned by the ``audit_wiki`` build tool."""

    passed: bool
    profile: str
    entity_count: int
    confirmed_count: int
    draft_count: int
    deprecated_count: int
    error_count: int
    warning_count: int
    issue_counts: dict[str, int]
    filtered_issue_count: int
    returned_issue_count: int
    next_offset: int
    snapshot_id: str
    source_snapshot_id: NotRequired[str]
    relation_coverage: dict[str, CoverageSummary]
    confirmed_candidates: list[str]
    issues: list[QualityIssue]


@dataclass(frozen=True, slots=True)
class RequirementCheck:
    """One structured or narrative relationship requirement."""

    code: str
    complete: bool
    message: str
    disposition: IssueDisposition = IssueDisposition.GAP
    opa_reason_code: str = "content_missing"
    # Concepts the requirement links to (e.g. ``("Symptom",)`` for
    # ``Fault.symptom.body_link``). When any target concept is already
    # materialized in the library, a failing check means "build the missing
    # link" (repair), not "content is missing from source" (permanent gap).
    target_concepts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AuditIssuePolicy:
    """Workflow policy attached to a stable audit issue code."""

    disposition: IssueDisposition
    opa_reason_code: str = ""


_REPAIR_ONLY = AuditIssuePolicy(IssueDisposition.REPAIR_ONLY)
_REFERENCE_GAP = AuditIssuePolicy(IssueDisposition.GAP, "content_missing")
_SOURCE_GAP = AuditIssuePolicy(IssueDisposition.GAP, "content_missing")
_CONTENT_GAP = AuditIssuePolicy(IssueDisposition.GAP, "content_missing")
_FACT_CONFLICT = AuditIssuePolicy(IssueDisposition.CONFLICT, "fact_conflict")

AUDIT_ISSUE_POLICIES: dict[str, AuditIssuePolicy] = {
    "entity_target_missing": _REPAIR_ONLY,
    "unconfirmed_entity": _REPAIR_ONLY,
    "dangling_relation_target": _REFERENCE_GAP,
    "typed_relation_wrong_concept": _FACT_CONFLICT,
    "dangling_wiki_reference": _REFERENCE_GAP,
    "Procedure.specification_ref_unresolvable": _REFERENCE_GAP,
    "Profile.direct_component_not_in_device_bom": _FACT_CONFLICT,
    "Profile.parent_symptom_not_indexed_by_device": _REFERENCE_GAP,
    "source_unresolvable": _SOURCE_GAP,
    "unresolved_placeholder": _REPAIR_ONLY,
    "dangling_reference": _REFERENCE_GAP,
    "malformed_uri": _REPAIR_ONLY,
    "Component.working_mechanism": _CONTENT_GAP,
    "Fault.failure_mechanism": _CONTENT_GAP,
    "empty_wiki": _REPAIR_ONLY,
}


def audit_issue_policy(code: str) -> AuditIssuePolicy | None:
    """Return the explicit workflow policy for one audit issue code.

    Validation hooks are deterministic write-format checks and therefore
    share one namespace-level policy.  All other codes must be registered
    explicitly or supplied by the producer (for example a
    :class:`RequirementCheck`).  Unknown codes deliberately return ``None``
    so discovery fails closed instead of inventing a gap category.
    """
    if code.startswith("hook."):
        return _REPAIR_ONLY
    return AUDIT_ISSUE_POLICIES.get(code)


def parse_frontmatter(content: str) -> dict[str, object]:
    """Parse top-level YAML frontmatter without accepting scalar roots."""
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end = next(
        (index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"),
        None,
    )
    if end is None:
        return {}
    try:
        loaded = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {key: value for key, value in loaded.items() if isinstance(key, str)}


def extract_sections(content: str) -> dict[str, str]:
    """Return ``##`` sections after stripping YAML frontmatter."""
    body = content
    lines = content.splitlines()
    if lines and lines[0].strip() == "---":
        end = next(
            (index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"),
            None,
        )
        if end is not None:
            body = "\n".join(lines[end + 1 :])

    matches = list(_SECTION_RE.finditer(body))
    return {
        match.group(1).strip(): body[
            match.end() : matches[index + 1].start() if index + 1 < len(matches) else len(body)
        ].strip()
        for index, match in enumerate(matches)
    }


def has_usable_procedure_criteria(section: str) -> bool:
    """Return whether a Procedure criteria section contains a real assertion.

    A page-level raw citation or text such as ``见来源`` is provenance, not a
    pass/fail standard.  The function intentionally does not require a
    numeric value: qualitative criteria such as ``无泄漏`` are valid too.
    """
    return _section_has_substance(section, min_chars=0)


# Placeholder phrase detector (pattern centralized in section_constants).
_PLACEHOLDER_TEXT_RE = re.compile(PLACEHOLDER_TEXT_RE)


def _section_has_substance(section: str, *, min_chars: int = 10) -> bool:
    """Return whether a body section has substantive explanatory content.

    Strips URIs and punctuation, rejects pure placeholder phrases, and
    optionally enforces a minimum character count on the remainder.
    """
    without_uris = re.sub(r"[a-z][a-z0-9+.\-]*://[^\s)\]>]+", "", section)
    normalized = re.sub(r"[\s\[\]（）()<>:：,，。；;、|#*_\-]+", "", without_uris)
    if not normalized:
        return False
    if min_chars and len(normalized) < min_chars:
        return False
    return not _PLACEHOLDER_TEXT_RE.match(normalized)


def extract_wiki_uris(content: str) -> set[str]:
    """Extract complete wiki URIs, including readable paths with parentheses.

    Markdown's closing ``)`` is treated as a delimiter only when it is not
    balancing an opening ASCII parenthesis inside the URI.  This avoids the
    historical URI truncation that turned a valid path such as ``主泵(变量)``
    into a dangling prefix.
    """
    result: set[str] = set()
    for uri in extract_source_uris(content):
        remainder = uri.removeprefix(_wiki_root_uri + "/")
        path, _, fragment = remainder.partition("#")
        parts = path.split("/")
        if len(parts) < 2 or not all(parts) or parts[0] not in _WIKI_CONCEPTS:
            continue
        if fragment and any(character.isspace() for character in fragment):
            continue
        result.add(uri)
    return result


def all_relation_uris(content: str, concept: str, field: str, root_uri: str = "") -> set[str]:
    """Get URIs for a relation field from frontmatter AND body section.

    Checks frontmatter first; also extracts ``viking://`` URIs from the
    corresponding body section per ``_BODY_LINK_MAP``.  Returns the
    combined set (fragment-anchors stripped).
    """
    uris: set[str] = set()
    frontmatter = parse_frontmatter(content)

    # From frontmatter
    fm_value = frontmatter.get(field)
    prefix = root_uri.rstrip("/") + "/" if root_uri else ""
    if isinstance(fm_value, str) and fm_value.strip():
        if not prefix or fm_value.startswith(prefix):
            uris.add(fm_value.split("#", 1)[0])
    elif isinstance(fm_value, list):
        for item in fm_value:
            if isinstance(item, str) and item.strip() and (not prefix or item.startswith(prefix)):
                uris.add(item.split("#", 1)[0])

    # From body section (fallback for fields in _BODY_LINK_MAP)
    body_key = "_Profile" if _is_profile(content) else concept
    field_info = _BODY_LINK_MAP.get(body_key, {}).get(field)
    if field_info:
        heading, _ = field_info
        sections = extract_sections(content)
        section_text = sections.get(heading, "")
        uris.update(u.split("#", 1)[0] for u in extract_wiki_uris(section_text))

    return uris


def extract_malformed_wiki_uris(content: str) -> list[str]:
    """Return class-only / placeholder ``{root_uri}/Concept/`` tails.

    Empty tail (missing Object segment) or placeholder templates
    (``<...>`` / ``{...}``) are malformed; readable two-segment URIs pass.
    """
    malformed: list[str] = []
    for uri in extract_source_uris(content):
        remainder = uri.removeprefix(_wiki_root_uri + "/")
        path = remainder.partition("#")[0]
        parts = path.split("/")
        if not parts or parts[0] not in _WIKI_CONCEPTS:
            continue
        if (
            len(parts) < 2
            or not all(parts[1:])
            or any(part.startswith(("<", "{")) for part in parts[1:])
            # ``open_gap: ...`` is body prose, not a navigable entity URI.
            # Treating it as a two-segment URI creates dangling references
            # that cannot be resolved or repaired deterministically.
            or any(part.startswith("open_gap:") for part in parts[1:])
        ):
            malformed.append("/".join(parts[1:]))
    return malformed


def extract_source_uris(content: str) -> set[str]:
    """Extract complete source URIs with Markdown-aware delimiters.

    Recognizes ``viking://`` and ``file://`` tokens.  Readable entity URIs may
    contain spaces in their object name.  A space is still a delimiter in
    prose, but it is part of a Markdown link destination
    (``[label](viking://...)``) or a quoted YAML scalar.  The old lexer
    truncated those destinations at the first space and produced false
    dangling links.
    """
    result: set[str] = set()
    # Raw chapter URIs may contain spaces (CJK names like "01_1 前言").
    # _RAW_CHAPTER_URI_RE handles these; match first so the space-delimited
    # lexer below skips the overlapping region instead of truncating at the space.
    chapter_spans: list[tuple[int, int]] = []
    for m in _RAW_CHAPTER_URI_RE.finditer(content):
        result.add(m.group())
        chapter_spans.append((m.start(), m.end()))
    for start_match in _URI_START_RE.finditer(content):
        if any(s <= start_match.start() < e for s, e in chapter_spans):
            continue
        index = start_match.start()
        cursor = start_match.end()
        parentheses = 0
        preceding = content[index - 1] if index else ""
        markdown_destination = preceding == "("
        quoted_scalar = preceding if preceding in {'"', "'"} else ""
        while cursor < len(content):
            character = content[cursor]
            if quoted_scalar:
                if character == quoted_scalar or character in "\r\n":
                    break
                cursor += 1
                continue
            if markdown_destination:
                if character == "(":
                    parentheses += 1
                elif character == ")":
                    if parentheses == 0:
                        break
                    parentheses -= 1
                cursor += 1
                continue
            if character.isspace() or character in "\"'<>[]":
                break
            if character == "(":
                parentheses += 1
            elif character == ")":
                if parentheses == 0:
                    break
                parentheses -= 1
            cursor += 1
        token = content[index:cursor].rstrip(_URI_TRAILING_PUNCTUATION)
        if token:
            result.add(token)
    return result


def has_unresolved_placeholder(content: str) -> bool:
    """Return whether agent TODO language remains in formal content."""
    return _UNRESOLVED_PLACEHOLDER_RE.search(content) is not None


def entity_status(content: str) -> str:
    """Return the normalized frontmatter status."""
    status = parse_frontmatter(content).get("status", "")
    return status.strip() if isinstance(status, str) else ""


def force_confirmed_status(content: str) -> str:
    """Return a validation-only copy whose status is ``confirmed``."""
    if re.search(r"^status\s*:", content, re.MULTILINE):
        return re.sub(
            r"^status\s*:.*$",
            "status: confirmed",
            content,
            count=1,
            flags=re.MULTILINE,
        )
    if content.startswith("---\n"):
        return content.replace("---\n", "---\nstatus: confirmed\n", 1)
    return content


def confirmation_requirements(
    content: str,
    concept: str,
    class_name: str = "",
) -> tuple[RequirementCheck, ...]:
    """Evaluate graph edges required for end-to-end diagnostic traversal."""
    frontmatter = parse_frontmatter(content)
    sections = extract_sections(content)
    checks: list[RequirementCheck] = []

    def field_present(
        code: str, field: str, message: str, *, target_concepts: tuple[str, ...] = ()
    ) -> None:
        fm_present = _has_value(frontmatter.get(field)) or bool(
            all_relation_uris(content, concept, field)
        )
        checks.append(
            RequirementCheck(
                code=code,
                complete=fm_present,
                message=message,
                disposition=IssueDisposition.GAP,
                opa_reason_code="content_missing",
                target_concepts=target_concepts,
            ),
        )

    if _has_value(frontmatter.get("profile_id")):
        return tuple(checks)

    if concept == "Device":
        field_present(
            "Device.critical_components",
            "critical_components",
            "设备必须引用其关重件清单。",
            target_concepts=("Component",),
        )
    elif concept == "Component":
        # Component associations are commonly owned by Device, Fault and
        # Procedure pages.  They are validated corpus-wide through
        # backlinks/graph isolation rather than by forcing empty lists onto
        # every Component page.
        # Working mechanism is the terminal node of the diagnostic chain:
        # Symptom → Fault → Component → 工作机理.  Without substantive
        # content here the chain dead-ends and the OPA is actionable.
        working = sections.get(SECTION_MECHANISM, "")
        checks.append(
            RequirementCheck(
                code="Component.working_mechanism",
                complete=_section_has_substance(working),
                message="部件必须包含工作机理章节的实质性内容，描述其工作原理。",
                disposition=IssueDisposition.GAP,
                opa_reason_code="content_missing",
            ),
        )
    elif concept == "DTC":
        field_present(
            "DTC.related_faults",
            "related_faults",
            "DTC 必须至少引用一个故障实体。",
            target_concepts=("Fault",),
        )
    elif concept == "Fault":
        # Failure mechanism is the core diagnostic content: without it
        # the page is a label, not an explanation of how the part fails.
        failure_mechanism = sections.get(SECTION_FAILURE_MECHANISM, "")
        checks.append(
            RequirementCheck(
                code="Fault.failure_mechanism",
                complete=_section_has_substance(failure_mechanism),
                message="故障必须包含失效机理章节的实质性内容，描述部件以何种方式失效。",
                disposition=IssueDisposition.GAP,
                opa_reason_code="content_missing",
            ),
        )

    return tuple(checks)


def _has_value(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return value is not None and value is not False


def _has_concept_uri(content: str, concept: str) -> bool:
    return (
        re.search(
            rf"{re.escape(_wiki_root_uri)}/{re.escape(concept)}/[^/\s()<>#]+",
            content,
        )
        is not None
    )


def is_optional_relation_field(concept: str, field: str) -> bool:
    """True when a schema relation field has no ``required: true`` marker.

    Unknown concept or field defaults to False (treated as required) so
    callers never defer relation work on schema shapes they cannot see.
    """
    try:
        schema = get_concept_schema(concept)
    except KeyError:
        return False
    frontmatter = schema.get("frontmatter")
    if not isinstance(frontmatter, list):
        return False
    for item in frontmatter:
        if isinstance(item, dict) and item.get("name") == field:
            return not bool(item.get("required"))
    return False


def is_optional_relation_issue(code: str, concept: str) -> bool:
    """True when a relationship_completeness audit flags only optional relations.

    Under the backbone policy only ``required: true`` schema relation
    fields are publication-critical; when a concept declares none, the
    issue is deferred instead of opening an OPA.
    """
    if code != "relationship_completeness":
        return False
    try:
        schema = get_concept_schema(concept)
    except KeyError:
        return False
    frontmatter = schema.get("frontmatter")
    if not isinstance(frontmatter, list):
        return False
    return not any(
        item.get("type") in {"ref", "ref_list"} and item.get("required")
        for item in frontmatter
        if isinstance(item, dict)
    )
