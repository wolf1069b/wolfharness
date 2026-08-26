"""TicketEngine — standalone OPA/OPS/OPL ticket engine for wiki knowledge revisions.

Extracted from ``opa.py`` (OPAMixin).  OPA records live under ``OP/OpA/``,
reference at least one KB URI, and do not register a natural key (so they
never enter the finalize/audit entity index).

This module is self-contained: it depends only on ``wolfharness.wiki.*``
(storage, models, quality, namespaces) and has no import on
``xeno_adp_agentic``.  When ``xeno_adp_agentic`` is available,
``WikiBuildTools`` can subclass or compose this engine to provide the
full entity-write path (``merge_entity``, ``_apply_operations``, etc.).
"""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
import logging
from pathlib import Path
import re
from typing import TYPE_CHECKING

from wolfharness.capabilities.wiki._helpers import _entity_batch_limit
from wolfharness.capabilities.wiki.models import (
    OPA_CLOSURE_STATUSES,
    OPA_REASON_CODES,
    OPAModel,
    OPLModel,
    OPSModel,
    infer_opa_reason_code,
)
from wolfharness.capabilities.wiki.quality import (
    BuildProfile,
    IssueDisposition,
    SourceReadResult,
    SourceReadStatus,
    audit_issue_policy,
    classify_raw_source_uri,
    extract_sections,
    extract_source_uris,
    is_optional_relation_issue,
    parse_frontmatter,
)


if TYPE_CHECKING:
    from wolfharness.capabilities.wiki.storage import WikiStore


logger = logging.getLogger(__name__)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_OPA_CATEGORY_DIRS: dict[str, str] = {
    "gap": "内容缺失",
    "conflict": "知识冲突",
    "feedback": "专家反馈",
}

_DEDUPE_KEY_MIN_LEN = 6
_MAX_RETRIEVAL_LIMIT = 50
_MAX_REPORT_LIMIT = 10000
_MAX_DISCOVERY_LIMIT = 500

# Relation/link gap codes that should NOT generate OPAs.  These are
# relationship completeness issues (dangling refs, missing links, isolated
# entities) that are best fixed via rebuild_index/backlinks, not via the
# manual OPA→OPS→OPL review pipeline.  Content-missing codes
# (source_missing, controller_identity, etc.) stay in OPA.
_RELATION_GAP_CODES: frozenset[str] = frozenset({
    # Policy-level relation codes
    "dangling_relation_target",
    "dangling_wiki_reference",
    "dangling_reference",
    "Procedure.specification_ref_unresolvable",
    "Profile.parent_symptom_not_indexed_by_device",
    "Profile.direct_component_not_in_device_bom",
    "relationship_completeness",
    # RequirementCheck relation-field codes (field_present with target_concepts)
    "SymptomProfile.device_refs",
    "SymptomProfile.direct_component_uri",
    "SymptomProfile.possible_faults",
    "Device.critical_components",
    "Device.symptom_refs",
    "DTC.related_faults",
    "Fault.affected_components",
    "Fault.procedures",
    "Procedure.target_components",
})

# Any code ending in ".body_link" is a section-link relation check.
_RELATION_GAP_SUFFIX = ".body_link"


def _is_relation_gap_code(code: str) -> bool:
    """True when an audit issue code is a relation/link gap, not a content gap."""
    return code in _RELATION_GAP_CODES or code.endswith(_RELATION_GAP_SUFFIX)


_SECTION_ANNOTATION_RE = re.compile(r"(?:section|章节)\s*[::]\s*([^\)\]\n,,]+)", re.IGNORECASE)

# Audit rules use stable machine field names while entity pages use Chinese
# section headings.  Keep this mapping in one place so discovery, audit
# downgrading, and finalize all agree on what it means for an OPA to be
# attached to a page section.
_SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "affected_components": ("影响范围",),
    "verification_procedures": ("验证方法",),
    "repair_procedures": ("修复方式",),
    "related_faults": ("可能失效机理",),
    "possible_faults": ("可能失效机理",),
    "assembly_parts": ("总成概览",),
    "target_components": ("操作目的",),
    "critical_components": ("关重件清单",),
}


def opa_section_names(target_section: str, annotations: str = "") -> set[str]:
    """Return machine and human names that can identify one page section."""
    names: set[str] = set()
    value = target_section.strip()
    if value:
        names.add(value)
        names.add(value.rsplit(".", 1)[-1])
        names.update(part.strip() for part in value.replace("|", "/").split("/") if part.strip())
        names.update(_SECTION_ALIASES.get(value.rsplit(".", 1)[-1], ()))
    names.update(
        match.strip() for match in _SECTION_ANNOTATION_RE.findall(annotations) if match.strip()
    )
    return names


class TicketEngine:
    """Standalone OPA/OPS/OPL ticket engine.

    Provides the full CRUD lifecycle for OPA/OPS/OPL records backed by
    a :class:`~wolfharness.wiki.storage.WikiStore`.  When
    ``xeno_adp_agentic`` is installed, ``WikiBuildTools`` can subclass
    or compose this engine and override the cross-mixin stubs below
    with full entity-write implementations.
    """

    def __init__(self, store: WikiStore) -> None:
        """Initialise the engine with a wiki store.

        Args:
            store: Wiki store used for all OPA/OPS/OPL persistence.
        """
        self.store = store
        # ponytail: _raw_fs mirrors the WikiBuildTools mixin contract;
        # use store._fs so raw-source URI prefix stripping works.
        self._raw_fs = store._fs

    # ------------------------------------------------------------------
    # Cross-mixin stubs — overridden by WikiBuildTools when available.
    # ------------------------------------------------------------------

    def read_resource(self, uri: str, line_numbers: bool = False) -> str:
        """Read a wiki entity page by URI.

        Strips the store root_uri prefix and delegates to ``store.read_text``.
        """
        key = uri
        root = self.store.root_uri
        if key.startswith(root + "/"):
            key = key[len(root) + 1 :]
        text = self.store.read_text(key)
        return text if text is not None else ""

    def read_raw_source(self, uri: str) -> SourceReadResult:
        """Read a raw source chapter.

        Stub: returns NOT_FOUND.  WikiBuildTools overrides with full
        raw-reader access.
        """
        return SourceReadResult(uri=uri, kind=None, status=SourceReadStatus.NOT_FOUND)

    def find_wiki(
        self,
        query: str,
        *,
        target_uri: str = "",
        limit: int = 10,
        deep: bool = False,
    ) -> list[dict[str, object]]:
        """Semantic search over wiki entities.

        Delegates to ``store._fs.find()`` when available (VikingFS),
        otherwise returns an empty list.
        """
        fs = self.store._fs
        finder = getattr(fs, "find", None)
        if finder is not None:
            result = finder(query, target_uri=target_uri, limit=limit, deep=deep)
            return list(result) if isinstance(result, list) else []
        return []

    def _apply_operations(self, content: str, operations: list[dict[str, object]]) -> str:
        """Apply patch operations to entity content.

        Stub: returns content unchanged.  The OPL ``candidate_content``
        field is the primary mechanism — operations are optional.
        """
        return content

    def merge_entity(
        self,
        concept: str,
        class_name: str,
        object_name: str,
        content: str,
        *,
        expected_sha256: str = "",
        **kwargs: object,
    ) -> str:
        """Write entity content and return its URI.

        Stub: writes directly via ``store.write_text_durable``.
        WikiBuildTools overrides with validation + hooks + sha256 lock.
        """
        key = f"{concept}/{class_name}/{object_name}.md"
        self.store.write_text_durable(key, content)
        return f"{self.store.root_uri}/{key}"

    def audit_wiki(self, **kwargs: object) -> dict[str, object]:
        """Run wiki audit and return issues.

        Stub: returns no issues.  WikiBuildTools overrides with full
        audit engine.
        """
        return {"issues": [], "next_offset": -1, "passed": True}

    def _opa_dir(self) -> Path:
        """Return the ``OP/OpA`` directory, creating it if absent."""
        path = self.store.root / self.store.CONCEPT_DIRS["OPA"]
        self.store.mkdir_p(self.store.CONCEPT_DIRS["OPA"])
        return path

    def _opa_dir_key(self) -> str:
        """Return the backend-relative OPA directory key."""
        key = self.store.CONCEPT_DIRS["OPA"]
        self.store.mkdir_p(key)
        return key

    def _target_hash(self, target_uri: str) -> str:
        """Return a stable 16-char directory segment for a target URI."""
        canonical = self.store.resolve_redirect(target_uri.strip()) if target_uri.strip() else ""
        return sha256(canonical.encode("utf-8")).hexdigest()[:16] if canonical else ""

    def _opa_key(self, opa_id: str, category: str = "", target_uri: str = "") -> str:
        """Return the backend-relative OPA key, optionally under a category and target subdir."""
        base = self._opa_dir_key()
        subdir = _OPA_CATEGORY_DIRS.get(category.strip().lower(), "")
        th = self._target_hash(target_uri)
        parts = [base]
        if subdir:
            parts.append(subdir)
        if th:
            parts.append(th)
        parts.append(f"{opa_id}.md")
        return "/".join(parts)

    def _find_opa_key(self, opa_id: str) -> str | None:
        """Locate an OPA by id, searching flat, category subdirs, and target-hash subdirs."""
        base = self._opa_dir_key()
        flat = f"{base}/{opa_id}.md"
        if self.store.read_text(flat) is not None:
            return flat
        for subdir in _OPA_CATEGORY_DIRS.values():
            key = f"{base}/{subdir}/{opa_id}.md"
            if self.store.read_text(key) is not None:
                return key
        for key in self.store.list_dir(base, recursive=True):
            if key.endswith(f"/{opa_id}.md") and self.store.read_text(key) is not None:
                return key
        return None

    def _op_dir_key(self, concept: str) -> str:
        """Return and create the storage key for one OP record type."""
        key = self.store.CONCEPT_DIRS[concept]
        self.store.mkdir_p(key)
        return key

    def _op_key(self, concept: str, record_id: str, target_uri: str = "") -> str:
        base = self._op_dir_key(concept)
        th = self._target_hash(target_uri)
        if th:
            return f"{base}/{th}/{record_id}.md"
        return f"{base}/{record_id}.md"

    def _find_op_key(self, concept: str, record_id: str) -> str | None:
        """Locate an OP record by id, trying flat path then target-hash subdirs."""
        base = self._op_dir_key(concept)
        flat = f"{base}/{record_id}.md"
        if self.store.read_text(flat) is not None:
            return flat
        for key in self.store.list_dir(base, recursive=True):
            if key.endswith(f"/{record_id}.md") and self.store.read_text(key) is not None:
                return key
        return None

    def _op_files(self, concept: str, *, target_uri: str = "") -> list[tuple[str, str]]:
        th = self._target_hash(target_uri)
        if th:
            subdir = f"{self._op_dir_key(concept)}/{th}"
            th_records = self._read_md_dir(subdir)
            if th_records:
                return sorted(th_records)
        records: list[tuple[str, str]] = []
        for key in self.store.list_dir(self._op_dir_key(concept), recursive=True):
            if not key.endswith(".md"):
                continue
            content = self.store.read_text(key)
            if content is not None:
                records.append((key, content))
        return sorted(records)

    def _read_md_dir(self, dir_key: str) -> list[tuple[str, str]]:
        records: list[tuple[str, str]] = []
        for key in self.store.list_dir(dir_key, recursive=False):
            if not key.endswith(".md"):
                continue
            content = self.store.read_text(key)
            if content is not None:
                records.append((key, content))
        return records

    def _opa_files(self, *, target_uri: str = "") -> list[tuple[str, str]]:
        """Read OPA Markdown files through local or Viking storage.

        When ``target_uri`` is given, only the matching target-hash
        subdir is read (O(1) directory listing); otherwise a full
        recursive scan is performed.
        """
        th = self._target_hash(target_uri)
        base = self._opa_dir_key()
        if th:
            records: list[tuple[str, str]] = []
            for subdir in _OPA_CATEGORY_DIRS.values():
                records.extend(self._read_md_dir(f"{base}/{subdir}/{th}"))
            return sorted(records)
        records: list[tuple[str, str]] = []
        for key in self.store.list_dir(base, recursive=True):
            if not key.endswith(".md"):
                continue
            content = self.store.read_text(key)
            if content is not None:
                records.append((key, content))
        return sorted(records)

    def _opa_uri(self, opa_id: str, category: str = "") -> str:
        """Return a URI that can actually be read from the active backend."""
        key = self._find_opa_key(opa_id)
        if key is None:
            key = self._opa_key(opa_id, category)
        return f"{self.store.root_uri}/{key}"

    def _op_uri(self, concept: str, record_id: str) -> str:
        if concept == "OPA":
            return self._opa_uri(record_id)
        key = self._find_op_key(concept, record_id)
        if key is not None:
            return f"{self.store.root_uri}/{key}"
        return f"{self.store.root_uri}/{self.store.CONCEPT_DIRS[concept]}/{record_id}.md"

    def is_valid_op_uri(self, uri: str) -> bool:
        """Return whether *uri* is an acceptable OPA/OPS reference URI.

        Mirrors the format rules enforced by ``_validate_opa_uris``: a
        ``viking://resources/...`` provider reference (any namespace), a URI
        inside the active wiki store's namespace, or a raw-source library
        URI all pass.  Existence is not checked -- the record may be filed
        before the entity is merged, and reference scopes may live outside
        the write store's namespace.
        """
        return (
            uri.startswith("viking://resources/")
            or self.store.is_wiki_uri(uri)
            or classify_raw_source_uri(uri, raw_root_uri=self._raw_fs.root_uri) is not None
        )

    def _validate_opa_uris(self, target_uri: str, evidence_uris: list[str]) -> None:
        """Require and validate the URI association of an OPA.

        Every OPA must reference at least one KB URI (the wiki entity under
        conflict or the source corpus that surfaced the conflict).  Formats
        are checked; existence is not (the record may be filed before the
        entity is merged).

        Accepts provider resource references outside the active write store.
        OPA/OPS records are written to the provider-owned ticket scope, while
        ``target_uri`` and ``evidence_uris`` are read/citation references and
        may point at a different provider-owned resource scope.
        """
        uris = [uri for uri in (target_uri, *evidence_uris) if uri]
        if not uris:
            raise ValueError(
                f"OPA must reference at least one URI: set target_uri "
                f"({self.store.root_uri}/...) or evidence_uris "
                f"(viking://resources/... source chapters).",
            )
        for uri in uris:
            if not self.is_valid_op_uri(uri):
                raise ValueError(
                    f"OPA URI must be a provider resource URI or local wiki/raw URI, got {uri!r}.",
                )

    @staticmethod
    def _list_value(value: object) -> list[str]:
        """Normalize a YAML list/scalar field into unique strings."""
        if isinstance(value, list):
            return list(
                dict[str, object].fromkeys(str(item).strip() for item in value if str(item).strip())
            )
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @staticmethod
    def _list_of_dicts(value: object) -> list[dict[str, object]]:
        """Read structured patch operations without accepting arbitrary YAML."""
        if not isinstance(value, list):
            return []
        return [dict[str, object](item) for item in value if isinstance(item, dict)]

    @staticmethod
    def _normalize_dedupe_text(value: object) -> str:
        """Normalize issue text so title/source wording does not create duplicates."""
        return re.sub(r"\s+", " ", str(value or "").strip()).casefold()

    def _opa_dedupe_key(
        self,
        *,
        category: str,
        reason_code: str,
        target_uri: str,
        target_section: str,
        finding: str,
        missing: str,
    ) -> str:
        """Return an evidence-independent identity for one logical OPA issue.

        A target section is the unit of work for audit and repair.  Multiple
        workers may describe that same section differently, so descriptions
        and evidence are deliberately not part of the primary key.  When no
        concrete target/section exists, the normalized finding remains the
        fallback discriminator.
        """
        target = self.store.resolve_redirect(target_uri.strip()) if target_uri.strip() else ""
        identity = [
            self._normalize_dedupe_text(category),
            self._normalize_dedupe_text(reason_code),
            target,
            self._normalize_dedupe_text(target_section),
        ]
        if not target or not target_section.strip():
            identity.extend((
                self._normalize_dedupe_text(finding),
                self._normalize_dedupe_text(missing),
            ))
        raw = "\x1f".join(identity)
        return "opa-dedupe-" + sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _canonical_uri_identity(self, value: str) -> str:
        """Return a redirect-stable identity while preserving URI fragments."""
        base, separator, fragment = value.partition("#")
        canonical = self.store.resolve_redirect(base) if self.store.is_wiki_uri(base) else base
        return canonical + (separator + fragment if separator else "")

    def _dedupe_uri_values(
        self, values: list[str], *, exclude: set[str] | None = None
    ) -> list[str]:
        excluded = {
            self._canonical_uri_identity(value.strip())
            for value in (exclude or set())
            if value.strip()
        }
        seen: set[str] = set()
        result: list[str] = []
        for raw_value in values:
            value = raw_value.strip()
            if not value:
                continue
            identity = self._canonical_uri_identity(value)
            if identity in excluded or identity in seen:
                continue
            seen.add(identity)
            result.append(value)
        return result

    @staticmethod
    def _merge_observations(previous: object, incoming: str) -> str:
        """Keep distinct observations while avoiding repeated generated prose."""
        separator = "\n\n--- additional observation ---\n\n"
        values = [part.strip() for part in str(previous or "").split(separator)]
        values.append(incoming.strip())
        seen: set[str] = set()
        unique: list[str] = []
        for value in values:
            if not value:
                continue
            identity = re.sub(r"[\s,。;;,.!?!?::]+", "", value).casefold()
            if identity in seen:
                continue
            seen.add(identity)
            unique.append(value)
        values = unique
        return "\n\n--- additional observation ---\n\n".join(values)

    def _find_opa_by_dedupe_key(
        self,
        dedupe_key: str,
        *,
        build_id: str = "",
    ) -> tuple[str, dict[str, object]] | None:
        """Find an existing OPA by its logical issue identity."""
        if not dedupe_key:
            return None
        for key, content in self._opa_files():
            frontmatter = self._record_from_content(content)
            if str(frontmatter.get("status", "")).strip().lower() in {"superseded", "rejected"}:
                continue
            if build_id and not self._opa_matches_build(key, frontmatter, build_id):
                continue
            if str(frontmatter.get("dedupe_key", "")).strip() == dedupe_key:
                return key, frontmatter
            # Legacy records did not persist a dedupe key.  Compute one from
            # their stable fields so the first new report consolidates them.
            category = str(frontmatter.get("category", ""))
            reason_code = str(frontmatter.get("reason_code", infer_opa_reason_code(category)))
            computed = self._opa_dedupe_key(
                category=category,
                reason_code=reason_code,
                target_uri=str(frontmatter.get("target_uri", "")),
                target_section=str(frontmatter.get("target_section", "")),
                finding=str(frontmatter.get("finding", "")),
                missing=str(frontmatter.get("missing", "")),
            )
            if computed == dedupe_key:
                return key, frontmatter
        return None

    def _opa_matches_build(self, key: str, frontmatter: dict[str, object], build_id: str) -> bool:
        """Match a record to a build without reviving unscoped old records."""
        record_build_id = str(frontmatter.get("build_id", "")).strip()
        if record_build_id:
            return record_build_id == build_id
        # Records without build_id were created before the checkpoint was
        # initialised (common early in a build).  Accept them as part of the
        # current build rather than silently dropping current-build OPAs —
        # the checkpoint build_id match is sufficient scoping.
        checkpoint = self.store.read_json("index/build_checkpoint.json")
        return (
            isinstance(checkpoint, dict) and str(checkpoint.get("build_id", "")).strip() == build_id
        )

    def _opa_category_from_key(self, key: str) -> str:
        for category, subdir in _OPA_CATEGORY_DIRS.items():
            if f"/{subdir}/" in f"/{key}":
                return category
        return ""

    @staticmethod
    def _human_key(title: str, category: str, target_section: str) -> str:
        """Create a compact readable display key for an OPA."""
        value = "__".join(
            part.strip() for part in (category, target_section, title) if part.strip()
        )
        value = re.sub(r"\s+", "-", value)
        value = re.sub(r"[\\/:*?\"<>|]+", "-", value)
        return value[:160].strip("-_") or "wiki-issue"

    @staticmethod
    def _clip_utf8(value: str, max_bytes: int) -> str:
        """Clip a string so its UTF-8 encoding fits in ``max_bytes``.

        Filenames are byte-length-limited (255 on disk, and Viking's unzip
        rejects longer paths), but the earlier character-based truncation
        let CJK ids reach ~570 bytes. Never split a multibyte codepoint.
        """
        encoded = value.encode("utf-8")
        if len(encoded) <= max_bytes:
            return value
        return encoded[:max_bytes].decode("utf-8", "ignore")

    @staticmethod
    def _readable_slug(value: str, *, fallback: str) -> str:
        """Keep identity names searchable without using opaque hashes."""
        slug = re.sub(r"\s+", "-", value.strip())
        slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "-", slug)
        slug = re.sub(r"-+", "-", slug).strip("-._")
        return TicketEngine._clip_utf8(slug, 60).rstrip("-._") or fallback

    @staticmethod
    def _record_from_content(content: str) -> dict[str, object]:
        """Parse an OPA record, recovering prose fields from the markdown body.

        Returns the frontmatter merged with:
        - ``title`` from the first H1 heading when the frontmatter lacks it
        - ``description`` / ``finding`` / ``missing`` / ``recommendation``
          from the body sections ``问题描述`` / ``冲突点`` / ``缺失点`` / ``建议``
          when the frontmatter lacks them (legacy YAML wins).
        """
        record = dict(parse_frontmatter(content or ""))
        if not str(record.get("title", "")).strip():
            match = re.search(r"^#\s+(.+)$", content or "", re.MULTILINE)
            if match is not None:
                record["title"] = match.group(1).strip()
        sections = extract_sections(content or "")
        for field, heading in (
            ("description", "问题描述"),
            ("finding", "冲突点"),
            ("missing", "缺失点"),
            ("recommendation", "建议"),
        ):
            if not str(record.get(field, "")).strip() and heading in sections:
                record[field] = sections[heading]
        return record

    def _readable_opa_id(
        self,
        *,
        category: str,
        reason_code: str,
        target_uri: str,
        target_path: str,
        target_section: str,
        title: str,
        dedupe_key: str = "",
    ) -> str:
        """Build a short, human-readable, dedupe-stable OPA filename.

        The readable target and title slugs keep the filename searchable,
        while the short dedupe suffix keeps it stable per logical issue
        without packing the whole record into the name.
        """
        target = target_uri.rstrip("/").rsplit("/", 1)[-1] if target_uri else (target_path or title)
        target_slug = self._readable_slug(target, fallback="target")[:20].rstrip("-._") or "target"
        title_slug = self._readable_slug(title, fallback="title")[:20].rstrip("-._") or "title"
        base = dedupe_key.removeprefix("opa-dedupe-") if dedupe_key else ""
        suffix = base[:8] if len(base) >= _DEDUPE_KEY_MIN_LEN else ""
        if not suffix:
            source = f"{category}\x1f{reason_code}\x1f{target_uri}\x1f{target_section}\x1f{title}"
            suffix = sha256(source.encode("utf-8")).hexdigest()[:8]
        kind = self._readable_slug(category, fallback="issue")[:12].rstrip("-._") or "issue"
        return self._clip_utf8(f"opa-{kind}-{target_slug}-{title_slug}-{suffix}", 80).rstrip("-._")

    @staticmethod
    def _opa_open_for_gate(record: dict[str, object]) -> bool:
        """Whether an OPA still blocks finalize, keyed on closure_status not status."""
        closure = str(record.get("closure_status", "")).strip().lower()
        status = str(record.get("status", "")).strip().lower()
        if closure == "closed":
            return False
        if closure == "deferred":
            return not str(record.get("closure_reason", "")).strip()
        if closure == "open":
            return True
        # Legacy record without closure_status: preserve prior gate semantics
        # so already-finalized builds are not retroactively blocked.
        return status == "pending"

    def _unresolved_opa_records(self, *, build_id: str | None = None) -> list[dict[str, object]]:
        """Return active OPA records for the selected build scope."""
        records: list[dict[str, object]] = []
        for key, content in self._opa_files():
            frontmatter = self._record_from_content(content)
            opa_id = Path(key).stem
            if str(frontmatter.get("status", "")).lower() not in {
                "resolved",
                "rejected",
                "superseded",
            }:
                record_build_id = str(frontmatter.get("build_id", "")).strip()
                if build_id is not None and not self._opa_matches_build(key, frontmatter, build_id):
                    continue
                records.append(
                    {
                        "opa_id": str(frontmatter.get("id", opa_id)),
                        "human_key": str(
                            frontmatter.get("human_key", frontmatter.get("title", opa_id))
                        ),
                        "uri": self._opa_uri(
                            str(frontmatter.get("id", opa_id)), self._opa_category_from_key(key)
                        ),
                        "status": str(frontmatter.get("status", "")),
                        "category": str(frontmatter.get("category", "")),
                        "reason_code": str(
                            frontmatter.get(
                                "reason_code",
                                infer_opa_reason_code(str(frontmatter.get("category", ""))),
                            ),
                        ),
                        "target_uri": str(frontmatter.get("target_uri", "")),
                        "target_section": str(frontmatter.get("target_section", "")),
                        "build_id": record_build_id,
                    },
                )
        return records

    def _is_explicit_tracked_record(self, record: dict[str, object]) -> bool:
        """Return whether a pending OPA is bound to a marked page section."""
        target_uri = str(record.get("target_uri", "")).strip()
        target_section = str(record.get("target_section", "")).strip()
        if not target_uri or not target_section:
            return False
        content = self.read_resource(target_uri)
        if content is None:
            return False
        sections = extract_sections(content)
        # discover_opa keeps a stable machine code in target_section (for
        # example ``Component.fault.body_link``), while the page uses a
        # human-readable heading. Accept the section annotation emitted in
        # the OPA description/finding so an explicit open_gap can downgrade
        # the corresponding required relation to a tracked warning.
        annotations = " ".join(
            str(record.get(field, ""))
            for field in ("description", "finding", "missing", "recommendation")
        )
        section_names = opa_section_names(target_section, annotations)
        matching_sections = [
            value
            for heading, value in sections.items()
            if any(name == heading or name in heading for name in section_names)
        ]
        category = str(record.get("category", "")).strip().lower()
        markers = (
            ("conflict_pending", "冲突")
            if category == "conflict"
            else ("open_gap", "待补充", "来源未说明", "来源缺失", "未物化")
        )
        if matching_sections:
            return any(
                any(marker in value.lower() for marker in markers) for value in matching_sections
            )
        frontmatter = parse_frontmatter(content)
        return target_section.rsplit(".", 1)[-1] in frontmatter and any(
            marker in content.lower() for marker in markers
        )

    def _is_explicit_gap_record(self, record: dict[str, object]) -> bool:
        """Backward-compatible alias for source-honest gap callers."""
        return str(
            record.get("category", "")
        ).strip().lower() == "gap" and self._is_explicit_tracked_record(record)

    def create_opa(
        self,
        *,
        title: str,
        description: str,
        category: str = "conflict",
        reason_code: str = "",
        scope: str = "entity",
        subtype: str = "wiki_error",
        target_uri: str = "",
        target_path: str = "",
        target_section: str = "",
        source_chapter: str = "",
        evidence_uris: list[str] | None = None,
        status: str = "pending",
        solution: str = "",
        opa_id: str = "",
        finding: str = "",
        missing: str = "",
        recommendation: str = "",
        related_uris: list[str] | None = None,
        dedupe_key: str = "",
        build_id: str = "",
        closure_status: str = "",
        closure_reason: str = "",
        skip_dedupe_lookup: bool = False,
    ) -> dict[str, object]:
        """结构化落盘一条 OPA 冲突/问题记录,替代手写 md。.

        存入 ``OP/OpA/<opa_id>.md``(YAML frontmatter + 证据 + 可解析 URI),
        不注册 natural key,因此不会进入 finalize/audit 的实体索引。

        ``opa_id`` 未提供时自动生成(同 target_uri + title 稳定复用)。
        返回 ``{opa_id, uri, path, title, target_uri}``。

        ``skip_dedupe_lookup`` 为 True 时跳过全目录去重检索——外部专家
        路径的 target_uri 是确定性输入,不需要扫全库找重复 OPA。
        """
        effective_category = category.strip().lower() or "conflict"
        effective_reason_code = reason_code or infer_opa_reason_code(effective_category)
        effective_target = target_uri.strip()
        if not build_id:
            checkpoint = self.store.read_json("index/build_checkpoint.json")
            if isinstance(checkpoint, dict):
                build_id = str(checkpoint.get("build_id", "")).strip()
        effective_section = target_section.strip()
        effective_finding = finding.strip()
        effective_missing = missing.strip()
        supplied_dedupe_key = dedupe_key.strip()
        dedupe_key = supplied_dedupe_key or self._opa_dedupe_key(
            category=effective_category,
            reason_code=effective_reason_code,
            target_uri=effective_target,
            target_section=effective_section,
            finding=effective_finding,
            missing=effective_missing,
        )
        existing_by_key: tuple[str, dict[str, object]] | None = None
        if not skip_dedupe_lookup:
            existing_by_key = self._find_opa_by_dedupe_key(dedupe_key, build_id=build_id)
        if existing_by_key is not None:
            # Dedupe is authoritative even when a caller supplies a readable
            # or legacy hash-like opa_id. Otherwise repeated merge/conflict
            # paths create parallel OPA records for one target section.
            opa_id = Path(existing_by_key[0]).stem
        elif not opa_id:
            opa_id = self._readable_opa_id(
                category=effective_category,
                reason_code=effective_reason_code,
                target_uri=effective_target,
                target_path=target_path,
                target_section=effective_section,
                title=title,
                dedupe_key=dedupe_key,
            )
        previous: dict[str, object] = {}
        existing_key = self._find_opa_key(opa_id)
        if existing_key and build_id:
            existing_content = self.store.read_text(existing_key)
            existing_frontmatter = self._record_from_content(existing_content or "")
            existing_build_id = str(existing_frontmatter.get("build_id", "")).strip()
            if existing_build_id != build_id:
                # Readable OPA IDs are intentionally stable within a build.
                # Add a build suffix on a cross-build collision so a new run
                # never mutates an old run's record.
                build_suffix = sha256(build_id.encode("utf-8")).hexdigest()[:10]
                opa_id = f"{opa_id}-b{build_suffix}"
                existing_key = self._find_opa_key(opa_id)
        existing = self.store.read_text(existing_key) if existing_key else None
        if existing is not None:
            previous = self._record_from_content(existing)
            dedupe_key = str(previous.get("dedupe_key", "")).strip() or dedupe_key
        allowed_statuses = {"pending", "resolved", "rejected", "superseded"}
        if status not in allowed_statuses:
            raise ValueError(f"Unsupported OPA status: {status!r}")
        if not category.strip():
            raise ValueError("OPA category must not be empty")
        previous_evidence = self._list_value(previous.get("evidence_uris"))
        previous_related = self._list_value(previous.get("related_uris"))
        effective_target = effective_target or str(previous.get("target_uri", ""))
        evidence = self._dedupe_uri_values([*previous_evidence, *list(evidence_uris or [])])
        if effective_target:
            evidence = [uri for uri in evidence if uri != effective_target]
        related = self._dedupe_uri_values(
            [*previous_related, *list(related_uris or [])],
            exclude=set(evidence) | ({effective_target} if effective_target else set()),
        )
        self._validate_opa_uris(effective_target, [*evidence, *related])
        effective_category = effective_category or str(previous.get("category", "conflict"))
        key = existing_key or self._opa_key(opa_id, effective_category, target_uri=effective_target)
        effective_reason_code = (
            reason_code
            or str(previous.get("reason_code", ""))
            or infer_opa_reason_code(effective_category)
        )
        if effective_reason_code not in OPA_REASON_CODES:
            raise ValueError(
                f"Unsupported OPA reason_code: {effective_reason_code!r}; "
                f"expected one of {', '.join(OPA_REASON_CODES)}",
            )
        effective_title = title or str(previous.get("title", opa_id))
        effective_description = self._merge_observations(
            previous.get("description", ""), description
        )
        human_key = str(previous.get("human_key", "")).strip() or self._human_key(
            effective_title,
            effective_category,
            target_section or str(previous.get("target_section", "")),
        )
        effective_section = effective_section or str(previous.get("target_section", ""))
        effective_finding = self._merge_observations(previous.get("finding", ""), effective_finding)
        effective_missing = self._merge_observations(previous.get("missing", ""), effective_missing)
        effective_recommendation = self._merge_observations(
            previous.get("recommendation", ""), recommendation
        )
        if status == "pending":
            required_fields = {
                "description": effective_description,
                "finding": effective_finding,
                "missing": effective_missing,
                "recommendation": effective_recommendation,
                # A readable target is already the OPA binding. Do not copy it
                # into evidence_uris just to satisfy the non-empty check.
                "evidence_uris": "\n".join(evidence) or effective_target,
            }
            missing_fields = [name for name, value in required_fields.items() if not value.strip()]
            if missing_fields:
                raise ValueError(
                    "pending OPA requires structured fields: " + ", ".join(missing_fields),
                )
        count_value: object = previous.get("report_count")
        report_count: int = (count_value if isinstance(count_value, int) else 0) + 1
        model = OPAModel(
            opa_id=opa_id,
            title=effective_title,
            description=effective_description,
            human_key=human_key,
            category=effective_category,
            reason_code=effective_reason_code,
            scope=scope or str(previous.get("scope", "entity")),
            subtype=subtype or str(previous.get("subtype", "wiki_error")),
            target_uri=effective_target,
            target_path=target_path or str(previous.get("target_path", "")),
            target_section=target_section or str(previous.get("target_section", "")),
            source_chapter=source_chapter or str(previous.get("source_chapter", "")),
            evidence_uris=evidence,
            status=status,
            solution=solution or str(previous.get("solution", "")),
            finding=effective_finding,
            missing=effective_missing,
            recommendation=effective_recommendation,
            related_uris=related,
            report_count=report_count,
            dedupe_key=dedupe_key or str(previous.get("dedupe_key", "")),
            build_id=build_id or str(previous.get("build_id", "")),
            closure_status=closure_status or str(previous.get("closure_status", "open")) or "open",
            closure_reason=closure_reason or str(previous.get("closure_reason", "")),
        )
        self.store.write_text_durable(
            key,
            model.to_markdown(ops_uri=self._op_uri("OPS", model.opa_id.replace("opa-", "ops-", 1))),
        )
        logger.info("OPA created: %s (target=%s)", opa_id, target_uri or target_path or "-")
        return {
            "opa_id": opa_id,
            "uri": self._opa_uri(opa_id, effective_category),
            "path": key,
            "title": effective_title,
            "human_key": human_key,
            "target_uri": effective_target,
            "reason_code": model.reason_code,
            "status": status,
            "closure_status": model.closure_status,
            "report_count": model.report_count,
            "dedupe_key": model.dedupe_key,
            "build_id": model.build_id,
        }

    @staticmethod
    def _record_id(value: str, prefix: str) -> str:
        """Accept either a record id or a backend URI and return its id."""
        token = value.rstrip("/").rsplit("/", 1)[-1].removesuffix(".md")
        return token if token.startswith(prefix + "-") else value

    def _opa_record(self, value: str) -> tuple[str, dict[str, object], str]:
        opa_id = self._record_id(value, "opa")
        key = self._find_opa_key(opa_id)
        if key is None:
            raise FileNotFoundError(f"OPA not found: {value}")
        content = self.store.read_text(key)
        if content is None:
            raise FileNotFoundError(f"OPA not found: {value}")
        category = ""
        for cat, subdir in _OPA_CATEGORY_DIRS.items():
            if f"/{subdir}/" in key:
                category = cat
                break
        return opa_id, parse_frontmatter(content), self._opa_uri(opa_id, category)

    def _validate_op_evidence(self, evidence_uris: list[str], related_uris: list[str]) -> None:
        self._validate_opa_uris("", [*evidence_uris, *related_uris])
        unreadable = [
            uri
            for uri in dict.fromkeys([*evidence_uris, *related_uris])
            if uri
            and not uri.startswith("viking://resources/")
            and self.read_resource(uri) is None
            and classify_raw_source_uri(uri, raw_root_uri=self._raw_fs.root_uri) is None
        ]
        if unreadable:
            raise ValueError(f"OP records require readable evidence/related URIs: {unreadable}")

    def _op_reference_quality(
        self,
        *,
        target_uri: str,
        evidence_uris: list[str],
        related_uris: list[str],
    ) -> list[str]:
        """Return hard quality violations for a generated OP reference set."""
        issues: list[str] = []
        target = target_uri.strip()
        evidence = [str(uri).strip() for uri in evidence_uris if str(uri).strip()]
        related = [str(uri).strip() for uri in related_uris if str(uri).strip()]

        def canonical(uri: str) -> str:
            return self.store.resolve_redirect(uri) if self.store.is_wiki_uri(uri) else uri

        evidence_keys = [canonical(uri) for uri in evidence]
        related_keys = [canonical(uri) for uri in related]
        target_key = canonical(target) if target else ""
        if len(evidence_keys) != len(set(evidence_keys)) or len(related_keys) != len(
            set(related_keys)
        ):
            issues.append("REFERENCE_DUPLICATE")
        if target_key and (target_key in evidence_keys or target_key in related_keys):
            issues.append("REFERENCE_TARGET_REPEATED")
        if set(evidence_keys) & set(related_keys):
            issues.append("REFERENCE_EVIDENCE_REPEATED_IN_RELATED")
        try:
            self._validate_opa_uris(target, [*evidence, *related])
        except ValueError:
            issues.append("REFERENCE_URI_INVALID")
        unreadable: list[str] = []
        for uri in dict[str, object].fromkeys([target, *evidence, *related]):
            if not uri:
                continue
            if self.store.is_wiki_uri(uri):
                readable = self.read_resource(uri) is not None
            else:
                raw_result = self.read_raw_source(uri)
                readable = raw_result.status is SourceReadStatus.OK
            if not readable:
                unreadable.append(uri)
        if unreadable:
            issues.append("REFERENCE_UNREADABLE")
        return list(dict[str, object].fromkeys(issues))

    def _require_parent_target(self, *, parent_target: str, requested_target: str) -> str:
        """Keep all records in an OP chain attached to the same target."""
        expected = parent_target.strip()
        requested = requested_target.strip()
        if requested and (
            not expected
            or self.store.resolve_redirect(requested) != self.store.resolve_redirect(expected)
        ):
            raise ValueError("target_uri must match parent OPA target_uri")
        return expected

    def _ops_retrieval_receipt(
        self,
        query: str,
        *,
        limit: int,
    ) -> tuple[list[str], list[str]]:
        """Run bounded retrieval over configured Wiki and raw roots."""
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("OPS requires a non-empty retrieval_query")
        if limit < 1 or limit > _MAX_RETRIEVAL_LIMIT:
            raise ValueError("OPS retrieval_limit must be between 1 and 50")
        scopes = [self.store.root_uri, self._raw_fs.root_uri]
        hit_uris: list[str] = []
        for scope in scopes:
            hits = self.find_wiki(
                normalized_query,
                target_uri=scope,
                limit=limit,
                deep=True,
            )
            hit_uris.extend(
                str(hit.get("uri", "")).strip() for hit in hits if str(hit.get("uri", "")).strip()
            )
        return scopes, list(dict[str, object].fromkeys(hit_uris))

    def _retrieval_scopes_match(self, scopes: list[str]) -> bool:
        """Accept the current raw root and the historical ``<root>/raw`` alias."""
        scope_set = set(scopes)
        raw_root = self._raw_fs.root_uri.rstrip("/")
        raw_aliases = {raw_root, raw_root + "/raw"}
        return self.store.root_uri in scope_set and bool(raw_aliases & scope_set)

    def create_ops(
        self,
        *,
        parent_opa: str,
        title: str,
        retrieval_query: str,
        retrieved_uris: list[str],
        analysis: str,
        solution: str,
        evidence_uris: list[str] | None = None,
        related_uris: list[str] | None = None,
        target_uri: str = "",
        status: str = "unconfirmed",
        ops_id: str = "",
        retrieval_limit: int = 10,
        candidate_content: str = "",
        candidate_operations: list[dict[str, object]] | None = None,
        expected_sha256: str = "",
        source_type: str = "pipeline",
    ) -> dict[str, object]:
        """Persist one source-backed expert recommendation for an OPA."""
        opa_id, opa_fm, opa_uri = self._opa_record(parent_opa)
        if status != "unconfirmed":
            raise ValueError("New OPS records must remain status=unconfirmed")
        candidate_operations = list(candidate_operations or [])
        if candidate_content.strip() and candidate_operations:
            raise ValueError("OPS must provide candidate_content or candidate_operations, not both")
        if any(not isinstance(operation, dict) for operation in candidate_operations):
            raise ValueError("OPS candidate_operations must contain only objects")
        if expected_sha256 and _SHA256_RE.fullmatch(expected_sha256) is None:
            raise ValueError("OPS expected_sha256 must be a 64-character lowercase SHA-256 hash")
        effective_target = self._require_parent_target(
            parent_target=str(opa_fm.get("target_uri", "")),
            requested_target=target_uri,
        )
        evidence = self._dedupe_uri_values(
            [*self._list_value(opa_fm.get("evidence_uris")), *(evidence_uris or [])],
            exclude={effective_target} if effective_target else set(),
        )
        related = self._dedupe_uri_values(
            [*self._list_value(opa_fm.get("related_uris")), *(related_uris or [])],
            exclude=set(evidence) | ({effective_target} if effective_target else set()),
        )
        self._validate_op_evidence(evidence, [effective_target, *related])
        target_is_readable = bool(
            effective_target and self.read_resource(effective_target) is not None
        )
        if (
            not analysis.strip()
            or not solution.strip()
            or (not evidence and not target_is_readable)
        ):
            raise ValueError(
                "OPS requires detailed analysis, solution, and at least one evidence URI"
            )
        retrieval_scopes, retrieval_hit_uris = self._ops_retrieval_receipt(
            retrieval_query,
            limit=retrieval_limit,
        )
        used_uris = list(
            dict[str, object].fromkeys(uri.strip() for uri in retrieved_uris if uri.strip())
        )
        missing_hits = [uri for uri in used_uris if uri not in retrieval_hit_uris]
        if missing_hits:
            raise ValueError(
                f"OPS retrieved_uris were not returned by the server-side "
                f"retrieval receipt: {missing_hits}",
            )
        # The formal target is implicitly bound to the OPS; it need not be
        # repeated in related_uris just to satisfy the retrieval receipt.
        unbound_hits = [
            uri for uri in used_uris if uri not in {*evidence, *related, effective_target}
        ]
        if unbound_hits:
            raise ValueError(
                f"OPS retrieved_uris must also appear in evidence_uris or "
                f"related_uris: {unbound_hits}",
            )
        if not ops_id:
            normalized_query = self._normalize_dedupe_text(retrieval_query)
            for existing_key, existing_content in self._op_files("OPS"):
                existing_fm = self._record_from_content(existing_content)
                existing_parent = str(existing_fm.get("parent_opa", ""))
                existing_target = str(existing_fm.get("target_uri", ""))
                if (
                    str(existing_fm.get("status", "unconfirmed")).lower() == "unconfirmed"
                    and existing_parent in {opa_uri, opa_id}
                    and self.store.resolve_redirect(existing_target)
                    == self.store.resolve_redirect(effective_target)
                    and self._normalize_dedupe_text(existing_fm.get("retrieval_query", ""))
                    == normalized_query
                ):
                    ops_id = Path(existing_key).stem
                    break
            if not ops_id:
                title_slug = (
                    self._readable_slug(title, fallback="recommendation")[:20].rstrip("-._")
                    or "recommendation"
                )
                ops_id = self._clip_utf8(
                    "ops-{}-{}-{}".format(
                        opa_id.removeprefix("opa-")[:20],
                        title_slug,
                        sha256(f"{opa_uri}\x1f{effective_target}\x1f{title}".encode()).hexdigest()[
                            :8
                        ],
                    ),
                    80,
                ).rstrip("-._")
        key = self._op_key("OPS", ops_id, target_uri=effective_target)
        previous = self._record_from_content(self.store.read_text(key) or "")
        model = OPSModel(
            ops_id=ops_id,
            parent_opa=opa_uri,
            title=title or str(previous.get("title", ops_id)),
            status=status,
            target_uri=effective_target,
            solution=self._merge_observations(previous.get("solution", ""), solution),
            analysis=self._merge_observations(previous.get("analysis", ""), analysis),
            retrieval_query=retrieval_query.strip() or str(previous.get("retrieval_query", "")),
            retrieval_scopes=retrieval_scopes,
            retrieval_hit_uris=list(
                dict[str, object].fromkeys(
                    [
                        *self._list_value(previous.get("retrieval_hit_uris")),
                        *retrieval_hit_uris,
                    ],
                ),
            ),
            # A retrieval receipt describes this invocation. Do not carry
            # stale used hits into a later retry where the evidence set may
            # have changed.
            retrieval_used_uris=used_uris,
            evidence_uris=self._dedupe_uri_values(
                [*self._list_value(previous.get("evidence_uris")), *evidence],
                exclude={effective_target} if effective_target else set(),
            ),
            related_uris=self._dedupe_uri_values(
                [*self._list_value(previous.get("related_uris")), *related],
                exclude=set(evidence) | ({effective_target} if effective_target else set()),
            ),
            candidate_content=candidate_content or str(previous.get("candidate_content", "")),
            candidate_operations=candidate_operations
            or self._list_of_dicts(previous.get("candidate_operations")),
            expected_sha256=expected_sha256 or str(previous.get("expected_sha256", "")),
            source_type=source_type or str(previous.get("source_type", "pipeline")),
            reviewed_by=str(previous.get("reviewed_by", "")),
            review_notes=str(previous.get("review_notes", "")),
            apply_status=str(previous.get("apply_status", "not_ready")),
            apply_error=str(previous.get("apply_error", "")),
            applied_at=str(previous.get("applied_at", "")),
            applied_entity_sha256=str(previous.get("applied_entity_sha256", "")),
        )
        self.store.write_text_durable(key, model.to_markdown())
        return {
            "ops_id": ops_id,
            "uri": self._op_uri("OPS", ops_id),
            "parent_opa": opa_uri,
            "target_uri": effective_target,
            "status": "unconfirmed",
            "retrieval_query": retrieval_query.strip(),
            "retrieval_scopes": retrieval_scopes,
            "retrieval_hit_uris": retrieval_hit_uris,
            "retrieval_used_uris": used_uris,
            "apply_status": model.apply_status,
        }

    def get_ops(
        self,
        *,
        parent_opa: str = "",
        status: str = "",
        target_uri: str = "",
        limit: int = 50,
    ) -> list[dict[str, object]]:
        """Read OPS recommendations, optionally scoped to one OPA/target."""
        parent_id = self._record_id(parent_opa, "opa") if parent_opa else ""
        parent_uri = ""
        if parent_id:
            try:
                _pid, _pfm, parent_uri = self._opa_record(parent_id)
            except FileNotFoundError:
                parent_uri = self._opa_uri(parent_id)
        out: list[dict[str, object]] = []
        for path, content in self._op_files("OPS", target_uri=target_uri):
            fm = self._record_from_content(content)
            record_parent = str(fm.get("parent_opa", ""))
            if parent_uri and record_parent not in {parent_uri, parent_id}:
                continue
            if status and str(fm.get("status", "")) != status:
                continue
            if target_uri and str(fm.get("target_uri", "")) != target_uri:
                continue
            ops_id = str(fm.get("id", Path(path).stem))
            out.append(
                {
                    "ops_id": ops_id,
                    "uri": self._op_uri("OPS", ops_id),
                    "path": path,
                    "parent_opa": record_parent,
                    "title": str(fm.get("title", ops_id)),
                    "status": str(fm.get("status", "unconfirmed")),
                    "target_uri": str(fm.get("target_uri", "")),
                    "retrieval_query": str(fm.get("retrieval_query", "")),
                    "retrieval_scopes": self._list_value(fm.get("retrieval_scopes")),
                    "retrieval_hit_uris": self._list_value(fm.get("retrieval_hit_uris")),
                    "retrieval_used_uris": self._list_value(fm.get("retrieval_used_uris")),
                    "evidence_uris": self._list_value(fm.get("evidence_uris")),
                    "related_uris": self._list_value(fm.get("related_uris")),
                    "candidate_content": str(fm.get("candidate_content", "")),
                    "candidate_operations": self._list_of_dicts(fm.get("candidate_operations")),
                    "expected_sha256": str(fm.get("expected_sha256", "")),
                    "source_type": str(fm.get("source_type", "pipeline")),
                    "reviewed_by": str(fm.get("reviewed_by", "")),
                    "review_notes": str(fm.get("review_notes", "")),
                    "apply_status": str(fm.get("apply_status", "not_ready")),
                    "apply_error": str(fm.get("apply_error", "")),
                    "applied_at": str(fm.get("applied_at", "")),
                    "applied_entity_sha256": str(fm.get("applied_entity_sha256", "")),
                },
            )
            if len(out) >= max(1, limit):
                break
        return out

    def ingest_external_ops(
        self,
        *,
        parent_opa: str = "",
        title: str,
        analysis: str,
        solution: str,
        evidence_uris: list[str],
        expert_id: str = "",
        expert_name: str = "",
        related_uris: list[str] | None = None,
        candidate_content: str = "",
        candidate_operations: list[dict[str, object]] | None = None,
        expected_sha256: str = "",
        ops_id: str = "",
        ops_uri: str = "",
    ) -> dict[str, object]:
        """Store external expert knowledge as an OPS draft, not an OPL.

        External input still requires business review.  It is therefore kept
        in the same editable OPS lifecycle as generated advice and cannot
        update formal entities until ``update_ops(status="confirmed")`` and
        an explicit ``apply_ops`` call.

        ``ops_uri`` lets an external expert revise an existing OPS record in
        place: the URI resolves to its ops_id, and repeated submissions with
        the same URI overwrite content (analysis / solution / evidence).
        ``parent_opa`` is only required on first creation; revisions inherit
        it from the existing record.
        """
        if not title.strip() or not analysis.strip() or not solution.strip():
            raise ValueError("External OPS requires title, analysis, and solution")
        if not evidence_uris:
            raise ValueError("External OPS requires a non-empty evidence_uris list")
        candidate_operations = list(candidate_operations or [])
        if candidate_content.strip() and candidate_operations:
            raise ValueError(
                "External OPS must provide candidate_content or candidate_operations, not both"
            )
        if any(not isinstance(operation, dict) for operation in candidate_operations):
            raise ValueError("External OPS candidate_operations must contain only objects")
        if expected_sha256 and _SHA256_RE.fullmatch(expected_sha256) is None:
            raise ValueError(
                "External OPS expected_sha256 must be a 64-character lowercase SHA-256 hash"
            )
        if not ops_id and ops_uri:
            ops_id = self._record_id(ops_uri, "ops")
        if ops_id:
            existing_key = self._find_op_key("OPS", ops_id)
            existing_content = self.store.read_text(existing_key) if existing_key else None
            if existing_content is not None:
                existing_fm = parse_frontmatter(existing_content)
                opa_uri = str(existing_fm.get("parent_opa", "") or parent_opa)
                target_uri = str(existing_fm.get("target_uri", ""))
                return self._rewrite_external_ops(
                    ops_id=ops_id,
                    parent_opa=opa_uri,
                    target_uri=target_uri,
                    title=title,
                    analysis=analysis,
                    solution=solution,
                    evidence_uris=evidence_uris,
                    related_uris=related_uris,
                    candidate_content=candidate_content,
                    candidate_operations=candidate_operations,
                    expected_sha256=expected_sha256,
                    expert_id=expert_id,
                    expert_name=expert_name,
                )
        if not parent_opa:
            raise ValueError("External OPS requires parent_opa when creating a new record")
        _opa_id, opa_fm, opa_uri = self._opa_record(parent_opa)
        target_uri = str(opa_fm.get("target_uri", ""))
        evidence = self._dedupe_uri_values(
            evidence_uris, exclude={target_uri} if target_uri else set()
        )
        related = self._dedupe_uri_values(
            related_uris or [],
            exclude=set(evidence) | ({target_uri} if target_uri else set()),
        )
        self._validate_op_evidence(evidence, [target_uri, *related])
        if not evidence and not (target_uri and self.read_resource(target_uri) is not None):
            raise ValueError("External OPS requires at least one readable evidence URI")
        if not ops_id:
            payload = "\x1f".join(
                (
                    opa_uri,
                    target_uri,
                    title,
                    expert_id,
                    candidate_content,
                    json.dumps(
                        candidate_operations,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            )
            ops_id = self._clip_utf8(
                "ops-external-"
                + self._readable_slug(title, fallback="expert")[:20].rstrip("-._")
                + "-"
                + sha256(payload.encode("utf-8")).hexdigest()[:10],
                100,
            ).rstrip("-._")
        key = self._find_op_key("OPS", ops_id)
        if key and self.store.read_text(key) is not None:
            return {
                "ops_id": ops_id,
                "uri": self._op_uri("OPS", ops_id),
                "parent_opa": opa_uri,
                "target_uri": target_uri,
                "status": "unconfirmed",
                "source_type": "external_expert",
                "apply_status": "not_ready",
                "idempotent": True,
            }
        model = OPSModel(
            ops_id=ops_id,
            parent_opa=opa_uri,
            title=title,
            status="unconfirmed",
            target_uri=target_uri,
            solution=solution,
            analysis=analysis,
            evidence_uris=evidence,
            related_uris=related,
            candidate_content=candidate_content,
            candidate_operations=candidate_operations,
            expected_sha256=expected_sha256,
            source_type="external_expert",
            reviewed_by="",
            review_notes=f"External expert: {expert_name or expert_id}",
            apply_status="not_ready",
        )
        result = self._write_ops_model(model)
        result.update({
            "source_type": "external_expert",
            "expert_id": expert_id,
            "expert_name": expert_name,
        })
        return result

    def _rewrite_external_ops(
        self,
        *,
        ops_id: str,
        parent_opa: str,
        target_uri: str,
        title: str,
        analysis: str,
        solution: str,
        evidence_uris: list[str],
        related_uris: list[str] | None,
        candidate_content: str,
        candidate_operations: list[dict[str, object]] | None,
        expected_sha256: str,
        expert_id: str,
        expert_name: str,
    ) -> dict[str, object]:
        """Overwrite content of an existing external OPS record in place."""
        evidence = self._dedupe_uri_values(
            evidence_uris, exclude={target_uri} if target_uri else set()
        )
        related = self._dedupe_uri_values(
            related_uris or [],
            exclude=set(evidence) | ({target_uri} if target_uri else set()),
        )
        self._validate_op_evidence(evidence, [target_uri, *related])
        model = OPSModel(
            ops_id=ops_id,
            parent_opa=parent_opa,
            title=title,
            status="unconfirmed",
            target_uri=target_uri,
            solution=solution,
            analysis=analysis,
            evidence_uris=evidence,
            related_uris=related,
            candidate_content=candidate_content,
            candidate_operations=list(candidate_operations or []),
            expected_sha256=expected_sha256,
            source_type="external_expert",
            reviewed_by="",
            review_notes=f"External expert: {expert_name or expert_id}",
            apply_status="not_ready",
        )
        result = self._write_ops_model(model)
        result.update({
            "source_type": "external_expert",
            "expert_id": expert_id,
            "expert_name": expert_name,
            "revised": True,
        })
        return result

    def _ops_record(self, value: str) -> tuple[str, str, dict[str, object]]:
        """Return an OPS id, storage key and parsed record."""
        ops_id = self._record_id(value, "ops")
        for path, content in self._op_files("OPS"):
            if Path(path).stem == ops_id:
                return ops_id, path, parse_frontmatter(content)
        raise FileNotFoundError(f"OPS not found: {value}")

    def _write_ops_model(self, model: OPSModel) -> dict[str, object]:
        """Persist an OPS draft/review state and return its canonical receipt."""
        key = self._op_key("OPS", model.ops_id, target_uri=str(model.target_uri))
        self.store.write_text_durable(key, model.to_markdown())
        return {
            "ops_id": model.ops_id,
            "uri": self._op_uri("OPS", model.ops_id),
            "parent_opa": model.parent_opa,
            "target_uri": model.target_uri,
            "status": model.status,
            "reviewed_by": model.reviewed_by,
            "apply_status": model.apply_status,
            "apply_error": model.apply_error,
        }

    def update_ops(
        self,
        ops_id: str,
        *,
        title: str | None = None,
        analysis: str | None = None,
        solution: str | None = None,
        evidence_uris: list[str] | None = None,
        related_uris: list[str] | None = None,
        candidate_content: str | None = None,
        candidate_operations: list[dict[str, object]] | None = None,
        expected_sha256: str | None = None,
        status: str | None = None,
        reviewed_by: str = "",
        review_notes: str = "",
    ) -> dict[str, object]:
        """Iterate or review one OPS draft without creating an OPL snapshot.

        ``unconfirmed`` is the machine-generated draft state.  A business
        reviewer may revise it and move it to ``confirmed`` or ``rejected``.
        Formal Wiki content is not changed by this method; use ``apply_ops``
        explicitly after confirmation.
        """
        current_id, _key, frontmatter = self._ops_record(ops_id)
        next_status = status or str(frontmatter.get("status", "unconfirmed"))
        if next_status not in {"unconfirmed", "confirmed", "rejected"}:
            raise ValueError("OPS status must be unconfirmed, confirmed, or rejected")
        if next_status in {"confirmed", "rejected"} and not (
            reviewed_by.strip() or str(frontmatter.get("reviewed_by", "")).strip()
        ):
            raise ValueError("Confirming or rejecting OPS requires reviewed_by")
        operations = (
            candidate_operations
            if candidate_operations is not None
            else self._list_of_dicts(frontmatter.get("candidate_operations"))
        )
        candidate = (
            candidate_content
            if candidate_content is not None
            else str(frontmatter.get("candidate_content", ""))
        )
        if candidate.strip() and operations:
            raise ValueError("OPS must provide candidate_content or candidate_operations, not both")
        next_sha = (
            expected_sha256
            if expected_sha256 is not None
            else str(frontmatter.get("expected_sha256", ""))
        )
        if next_sha and _SHA256_RE.fullmatch(next_sha) is None:
            raise ValueError("OPS expected_sha256 must be a 64-character lowercase SHA-256 hash")
        target_uri = str(frontmatter.get("target_uri", ""))
        evidence = self._dedupe_uri_values(
            evidence_uris
            if evidence_uris is not None
            else self._list_value(frontmatter.get("evidence_uris")),
            exclude={target_uri} if target_uri else set(),
        )
        related = self._dedupe_uri_values(
            related_uris
            if related_uris is not None
            else self._list_value(frontmatter.get("related_uris")),
            exclude=set(evidence) | ({target_uri} if target_uri else set()),
        )
        self._validate_op_evidence(evidence, [target_uri, *related])
        model = OPSModel(
            ops_id=current_id,
            parent_opa=str(frontmatter.get("parent_opa", "")),
            title=title if title is not None else str(frontmatter.get("title", current_id)),
            status=next_status,
            target_uri=target_uri,
            solution=solution if solution is not None else str(frontmatter.get("solution", "")),
            analysis=analysis if analysis is not None else str(frontmatter.get("analysis", "")),
            retrieval_query=str(frontmatter.get("retrieval_query", "")),
            retrieval_scopes=self._list_value(frontmatter.get("retrieval_scopes")),
            retrieval_hit_uris=self._list_value(frontmatter.get("retrieval_hit_uris")),
            retrieval_used_uris=self._list_value(frontmatter.get("retrieval_used_uris")),
            evidence_uris=evidence,
            related_uris=related,
            candidate_content=candidate,
            candidate_operations=list(operations),
            expected_sha256=next_sha,
            source_type=str(frontmatter.get("source_type", "pipeline")),
            reviewed_by=reviewed_by.strip() or str(frontmatter.get("reviewed_by", "")),
            review_notes=review_notes.strip() or str(frontmatter.get("review_notes", "")),
            apply_status="rejected"
            if next_status == "rejected"
            else str(frontmatter.get("apply_status", "not_ready")),
            apply_error=str(frontmatter.get("apply_error", "")),
            applied_at=str(frontmatter.get("applied_at", "")),
            applied_entity_sha256=str(frontmatter.get("applied_entity_sha256", "")),
        )
        result = self._write_ops_model(model)
        result["updated"] = True
        return result

    def apply_ops(self, ops_id: str) -> dict[str, object]:
        """Apply a confirmed OPS candidate with optimistic locking.

        This is deliberately separate from OPS creation and review.  No
        automatically generated draft can mutate a formal entity.
        """
        current_id, _key, frontmatter = self._ops_record(ops_id)
        if str(frontmatter.get("status", "unconfirmed")) != "confirmed":
            raise ValueError("Only confirmed OPS records can be applied")
        if str(frontmatter.get("apply_status", "")) == "applied":
            return {
                "ops_id": current_id,
                "uri": self._op_uri("OPS", current_id),
                "status": "confirmed",
                "apply_status": "applied",
                "idempotent": True,
            }
        target_uri = str(frontmatter.get("target_uri", ""))
        target_info = self.store.lookup_by_uri(target_uri)
        if target_info is None:
            raise ValueError("Confirmed OPS target_uri is not a known Wiki entity")
        concept, class_name, object_name = target_info[0], target_info[1] or "", target_info[2]
        candidate = str(frontmatter.get("candidate_content", ""))
        operations = self._list_of_dicts(frontmatter.get("candidate_operations"))
        if not candidate and not operations:
            return self._write_ops_model(
                OPSModel(
                    ops_id=current_id,
                    parent_opa=str(frontmatter.get("parent_opa", "")),
                    title=str(frontmatter.get("title", current_id)),
                    status="confirmed",
                    target_uri=target_uri,
                    solution=str(frontmatter.get("solution", "")),
                    analysis=str(frontmatter.get("analysis", "")),
                    retrieval_query=str(frontmatter.get("retrieval_query", "")),
                    retrieval_scopes=self._list_value(frontmatter.get("retrieval_scopes")),
                    retrieval_hit_uris=self._list_value(frontmatter.get("retrieval_hit_uris")),
                    retrieval_used_uris=self._list_value(frontmatter.get("retrieval_used_uris")),
                    evidence_uris=self._list_value(frontmatter.get("evidence_uris")),
                    related_uris=self._list_value(frontmatter.get("related_uris")),
                    apply_status="needs_review",
                    apply_error="Confirmed OPS has no candidate content or operations.",
                ),
            )
        current = self.store.read_entity(concept, class_name or None, object_name)
        expected_sha256 = str(frontmatter.get("expected_sha256", ""))
        if current is not None:
            current_sha256 = sha256(current.encode("utf-8")).hexdigest()
            if not expected_sha256:
                return self._update_ops_apply_status(
                    current_id,
                    "needs_review",
                    "Existing target requires expected_sha256 before application.",
                )
            if expected_sha256 != current_sha256:
                return self._update_ops_apply_status(
                    current_id, "failed", "Target changed; refresh OPS with a new expected_sha256."
                )
            if not candidate:
                candidate = self._apply_operations(current, operations)
        elif not candidate:
            return self._update_ops_apply_status(
                current_id, "failed", "Patch operations cannot create a missing entity."
            )
        try:
            entity_uri = self.merge_entity(
                concept,
                class_name,
                object_name,
                candidate,
                expected_sha256=expected_sha256,
                # Expert apply path: the confirmed record's own authority must
                # not revert its candidate (mirrors apply_opl).
                conflict_policy="external_authority",
            )
        except (FileNotFoundError, OSError, ValueError) as error:
            return self._update_ops_apply_status(current_id, "failed", str(error))
        applied_hash = sha256(
            (self.store.read_entity_by_uri(entity_uri) or "").encode("utf-8")
        ).hexdigest()
        result = self._update_ops_apply_status(
            current_id, "applied", "", applied_entity_sha256=applied_hash
        )
        result.update({"entity_uri": entity_uri})
        return result

    def _update_ops_apply_status(
        self,
        ops_id: str,
        apply_status: str,
        apply_error: str,
        *,
        applied_entity_sha256: str = "",
    ) -> dict[str, object]:
        """Update only application metadata while preserving the reviewed OPS."""
        current_id, _key, fm = self._ops_record(ops_id)
        model = OPSModel(
            ops_id=current_id,
            parent_opa=str(fm.get("parent_opa", "")),
            title=str(fm.get("title", current_id)),
            status=str(fm.get("status", "unconfirmed")),
            target_uri=str(fm.get("target_uri", "")),
            solution=str(fm.get("solution", "")),
            analysis=str(fm.get("analysis", "")),
            retrieval_query=str(fm.get("retrieval_query", "")),
            retrieval_scopes=self._list_value(fm.get("retrieval_scopes")),
            retrieval_hit_uris=self._list_value(fm.get("retrieval_hit_uris")),
            retrieval_used_uris=self._list_value(fm.get("retrieval_used_uris")),
            evidence_uris=self._list_value(fm.get("evidence_uris")),
            related_uris=self._list_value(fm.get("related_uris")),
            candidate_content=str(fm.get("candidate_content", "")),
            candidate_operations=self._list_of_dicts(fm.get("candidate_operations")),
            expected_sha256=str(fm.get("expected_sha256", "")),
            source_type=str(fm.get("source_type", "pipeline")),
            reviewed_by=str(fm.get("reviewed_by", "")),
            review_notes=str(fm.get("review_notes", "")),
            apply_status=apply_status,
            apply_error=apply_error,
            applied_at=datetime.now(UTC).isoformat()
            if apply_status == "applied"
            else str(fm.get("applied_at", "")),
            applied_entity_sha256=applied_entity_sha256 or str(fm.get("applied_entity_sha256", "")),
        )
        return self._write_ops_model(model)

    def create_opl(
        self,
        *,
        parent_opa: str,
        ops_uris: list[str],
        title: str,
        proposal: str,
        rationale: str,
        evidence_uris: list[str] | None = None,
        related_uris: list[str] | None = None,
        target_uri: str = "",
        status: str = "unconfirmed",
        opl_id: str = "",
        opl_uri: str = "",
        source_type: str = "pipeline",
        expert_id: str = "",
        expert_name: str = "",
        candidate_content: str = "",
        candidate_operations: list[dict[str, object]] | None = None,
        expected_sha256: str = "",
        archive_reason: str = "",
    ) -> dict[str, object]:
        """Form an unconfirmed knowledge proposal from one OPA and its OPS.

        OPL is an optional derived snapshot, not a fact source.  Automatic
        builds must not create OPL records: only an explicit manual archive
        (``archive_reason``) may persist one, and every referenced OPS must
        already be ``confirmed`` — an unconfirmed OPS means the proposal is
        still being reviewed and cannot be archived yet.
        """
        opa_id, opa_fm, opa_uri = self._opa_record(parent_opa)
        if status != "unconfirmed":
            raise ValueError("New OPL records must remain status=unconfirmed")
        if not ops_uris:
            raise ValueError("OPL requires at least one OPS URI")
        ops_rows = self.get_ops(parent_opa=opa_id, limit=200)
        ops_by_uri = {str(row["uri"]): row for row in ops_rows}
        missing_ops = [uri for uri in ops_uris if uri not in ops_by_uri]
        if missing_ops:
            raise ValueError(f"OPL references OPS records not attached to {opa_id}: {missing_ops}")
        if source_type != "external_expert" and not archive_reason.strip():
            raise ValueError(
                "OPL is a derived snapshot, not a fact source: automatic builds must not "
                "create OPL records. Provide archive_reason to archive an explicit "
                "manual snapshot.",
            )
        unconfirmed_ops = [
            uri
            for uri in ops_uris
            if str(ops_by_uri[uri].get("status", "unconfirmed")) != "confirmed"
        ]
        if unconfirmed_ops:
            raise ValueError(
                "OPL may only archive confirmed OPS records; unconfirmed OPS must be "
                f"reviewed (update_ops status=confirmed) first: {unconfirmed_ops}",
            )
        inherited_evidence = self._list_value(opa_fm.get("evidence_uris"))
        for row in ops_rows:
            if row["uri"] in ops_uris:
                inherited_evidence.extend(self._list_value(row["evidence_uris"]))
        effective_target = self._require_parent_target(
            parent_target=str(opa_fm.get("target_uri", "")),
            requested_target=target_uri,
        )
        evidence = self._dedupe_uri_values(
            [*inherited_evidence, *(evidence_uris or [])],
            exclude={effective_target} if effective_target else set(),
        )
        related = self._dedupe_uri_values(
            [*self._list_value(opa_fm.get("related_uris")), *(related_uris or [])],
            exclude=set(evidence) | ({effective_target} if effective_target else set()),
        )
        mismatched_ops = [
            uri
            for uri in ops_uris
            if self.store.resolve_redirect(str(ops_by_uri[uri].get("target_uri", "")))
            != self.store.resolve_redirect(effective_target)
        ]
        if mismatched_ops:
            raise ValueError(
                f"OPL references OPS records with a different target_uri: {mismatched_ops}"
            )
        self._validate_op_evidence(evidence, [effective_target, *related, *ops_uris])
        target_is_readable = bool(
            effective_target and self.read_resource(effective_target) is not None
        )
        if (
            not proposal.strip()
            or not rationale.strip()
            or (not evidence and not target_is_readable)
        ):
            raise ValueError("OPL requires detailed proposal, rationale, and evidence URIs")
        if not opl_id and opl_uri:
            opl_id = self._record_id(opl_uri, "opl")
        if not opl_id:
            requested_ops = set(
                dict[str, object].fromkeys(uri.strip() for uri in ops_uris if uri.strip())
            )
            for existing_key, existing_content in self._op_files("OPL"):
                existing_fm = parse_frontmatter(existing_content)
                existing_ops = set(self._list_value(existing_fm.get("ops_uris")))
                if (
                    str(existing_fm.get("status", "unconfirmed")).lower() == "unconfirmed"
                    and existing_ops == requested_ops
                    and self.store.resolve_redirect(str(existing_fm.get("target_uri", "")))
                    == self.store.resolve_redirect(effective_target)
                ):
                    opl_id = Path(existing_key).stem
                    break
            if not opl_id:
                title_slug = (
                    self._readable_slug(title, fallback="proposal")[:20].rstrip("-._") or "proposal"
                )
                opl_id = self._clip_utf8(
                    "opl-{}-{}-{}".format(
                        opa_id.removeprefix("opa-")[:20],
                        title_slug,
                        sha256(f"{opa_uri}\x1f{effective_target}\x1f{title}".encode()).hexdigest()[
                            :8
                        ],
                    ),
                    80,
                ).rstrip("-._")
        key = self._op_key("OPL", opl_id, target_uri=effective_target)
        previous = parse_frontmatter(self.store.read_text(key) or "")
        model = OPLModel(
            opl_id=opl_id,
            title=title or str(previous.get("title", opl_id)),
            parent_opa=opa_uri,
            ops_uris=list(
                dict[str, object].fromkeys(uri.strip() for uri in ops_uris if uri.strip())
            ),
            target_uri=effective_target,
            status="unconfirmed",
            proposal=self._merge_observations(previous.get("proposal", ""), proposal),
            rationale=self._merge_observations(previous.get("rationale", ""), rationale),
            evidence_uris=self._dedupe_uri_values(
                [*self._list_value(previous.get("evidence_uris")), *evidence],
                exclude={effective_target} if effective_target else set(),
            ),
            related_uris=self._dedupe_uri_values(
                [*self._list_value(previous.get("related_uris")), *related],
                exclude=set(evidence) | ({effective_target} if effective_target else set()),
            ),
            source_type=source_type,
            expert_id=expert_id,
            expert_name=expert_name,
            candidate_content=candidate_content,
            candidate_operations=list(candidate_operations or []),
            expected_sha256=expected_sha256,
            archive_reason=archive_reason.strip(),
        )
        self.store.write_text_durable(key, model.to_markdown())
        return {
            "opl_id": opl_id,
            "uri": self._op_uri("OPL", opl_id),
            "parent_opa": opa_uri,
            "ops_uris": list(dict[str, object].fromkeys(ops_uris)),
            "target_uri": effective_target,
            "status": "unconfirmed",
            "source_type": source_type,
            "apply_status": "not_applied" if source_type == "external_expert" else "not_applicable",
        }

    def _write_opl_model(self, model: OPLModel) -> dict[str, object]:
        """Persist an OPL model and return its canonical receipt."""
        key = self._op_key("OPL", model.opl_id, target_uri=str(model.target_uri))
        self.store.write_text_durable(key, model.to_markdown())
        return {
            "opl_id": model.opl_id,
            "uri": self._op_uri("OPL", model.opl_id),
            "parent_opa": model.parent_opa,
            "ops_uris": list(model.ops_uris),
            "target_uri": model.target_uri,
            "status": model.status,
            "source_type": model.source_type,
            "apply_status": model.apply_status,
            "apply_error": model.apply_error,
        }

    def _external_opl_id(
        self,
        *,
        target_uri: str,
        title: str,
        expert_id: str,
        candidate_content: str,
        candidate_operations: list[dict[str, object]],
    ) -> str:
        payload = "\x1f".join(
            (
                target_uri,
                title,
                expert_id,
                candidate_content,
                json.dumps(
                    candidate_operations, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
            ),
        )
        return self._clip_utf8(
            "opl-external-"
            + self._readable_slug(title, fallback="expert")[:20].rstrip("-._")
            + "-"
            + sha256(payload.encode("utf-8")).hexdigest()[:10],
            100,
        ).rstrip("-._")

    def ingest_external_opl(
        self,
        *,
        title: str,
        target_uri: str,
        proposal: str,
        rationale: str,
        evidence_uris: list[str],
        expert_id: str = "",
        expert_name: str = "",
        source_uri: str = "",
        related_uris: list[str] | None = None,
        candidate_content: str = "",
        candidate_operations: list[dict[str, object]] | None = None,
        target_concept: str = "",
        target_class_name: str = "",
        target_object_name: str = "",
        expected_sha256: str = "",
        opl_id: str = "",
        opl_uri: str = "",
        auto_apply: bool = True,
    ) -> dict[str, object]:
        """Ingest a direct expert OPL and optionally apply its machine patch.

        Direct expert input does not need a pre-existing OPS.  The method
        creates a resolved ``expert_feedback`` OPA as the audit parent, stores
        the OPL with its expert identity and evidence, then applies only a
        full candidate page or deterministic patch operations.  Application
        uses optimistic locking and an explicit external-authority policy;
        free-form prose is stored but never guessed into the Wiki.
        """
        if not title.strip() or not proposal.strip() or not rationale.strip():
            raise ValueError("External OPL requires title, proposal, and rationale")
        if not target_uri.strip():
            raise ValueError("External OPL requires target_uri")
        if not isinstance(evidence_uris, list) or not evidence_uris:
            raise ValueError("External OPL requires a non-empty evidence_uris list")
        if not (expert_id.strip() or expert_name.strip()):
            raise ValueError("External OPL requires expert_id or expert_name")
        candidate_operations = list(candidate_operations or [])
        if candidate_content.strip() and candidate_operations:
            raise ValueError(
                "External OPL must provide candidate_content or candidate_operations, not both"
            )
        if any(not isinstance(operation, dict) for operation in candidate_operations):
            raise ValueError("External OPL candidate_operations must contain only objects")
        if expected_sha256 and _SHA256_RE.fullmatch(expected_sha256) is None:
            raise ValueError(
                "External OPL expected_sha256 must be a 64-character lowercase SHA-256 hash"
            )
        target_info = self.store.lookup_by_uri(target_uri)
        if target_info is not None:
            target_concept = target_concept or target_info[0]
            target_class_name = target_class_name or target_info[1] or ""
            target_object_name = target_object_name or target_info[2]
        if not target_concept or not target_object_name:
            raise ValueError("External OPL requires a resolvable target_uri or target identity")
        expected_target_uri = self.store.entity_uri(
            target_concept,
            target_class_name or None,
            target_object_name,
        )
        if self.store.resolve_redirect(target_uri) != self.store.resolve_redirect(
            expected_target_uri
        ):
            raise ValueError(
                f"External OPL target does not match target identity: "
                f"{target_uri} != {expected_target_uri}",
            )
        current_content = self.store.read_entity(
            target_concept, target_class_name or None, target_object_name
        )
        target_is_readable = self.read_resource(target_uri) is not None
        evidence_candidates = [*evidence_uris, source_uri]
        effective_evidence = self._dedupe_uri_values(
            evidence_candidates,
            exclude={target_uri} if target_is_readable else set(),
        )
        if not effective_evidence:
            raise ValueError("External OPL requires at least one readable evidence URI")
        effective_related = self._dedupe_uri_values(
            related_uris or [],
            exclude=set(effective_evidence) | {target_uri},
        )
        if not opl_id and opl_uri:
            opl_id = self._record_id(opl_uri, "opl")
        if not opl_id:
            opl_id = self._external_opl_id(
                target_uri=target_uri,
                title=title,
                expert_id=expert_id,
                candidate_content=candidate_content,
                candidate_operations=candidate_operations,
            )
        opl_key = self._find_op_key("OPL", opl_id)
        if not opl_key:
            raise FileNotFoundError(f"OPL not found: {opl_id}")
        existing = self.store.read_text(opl_key)
        if existing is not None:
            previous = parse_frontmatter(existing)
            if str(previous.get("apply_status", "")) == "applied":
                return {
                    "opl_id": opl_id,
                    "uri": self._op_uri("OPL", opl_id),
                    "status": str(previous.get("status", "applied")),
                    "apply_status": "applied",
                    "idempotent": True,
                }
        previous_parent_opa = (
            parse_frontmatter(existing).get("parent_opa", "") if existing is not None else ""
        )
        if previous_parent_opa:
            parent_opa = str(previous_parent_opa)
        else:
            opa = self.create_opa(
                title=f"外部专家反馈:{title}",
                description=f"外部专家 {expert_name or expert_id} 提交了直接 OPL。",
                category="feedback",
                reason_code="expert_feedback",
                subtype="expert_feedback",
                target_uri=target_uri,
                target_section="external_opl",
                evidence_uris=effective_evidence,
                status="resolved",
                solution="已接收外部专家 OPL;正式实体更新状态见 OPL apply_status。",
                finding=proposal,
                missing="",
                recommendation="保留专家身份、来源和候选变更;仅对结构化候选内容执行幂等更新。",
                related_uris=effective_related,
            )
            parent_opa = str(opa["uri"])
        machine_candidate = bool(candidate_content.strip() or candidate_operations)
        apply_status = "pending" if auto_apply and machine_candidate else "needs_review"
        apply_error = ""
        if apply_status == "pending" and current_content is not None and not expected_sha256:
            apply_status = "needs_review"
            apply_error = "Existing target requires expected_sha256 before automatic application."
        model = OPLModel(
            opl_id=opl_id,
            title=title,
            parent_opa=parent_opa,
            target_uri=target_uri,
            status="unconfirmed",
            proposal=proposal,
            rationale=rationale,
            evidence_uris=effective_evidence,
            related_uris=effective_related,
            source_type="external_expert",
            expert_id=expert_id,
            expert_name=expert_name,
            source_uri=source_uri,
            target_concept=target_concept,
            target_class_name=target_class_name,
            target_object_name=target_object_name,
            expected_sha256=expected_sha256,
            candidate_content=candidate_content,
            candidate_operations=candidate_operations,
            apply_status=apply_status,
            apply_error=apply_error,
        )
        stored = self._write_opl_model(model)
        if apply_status != "pending":
            return stored
        return self.apply_opl(opl_id)

    def _update_external_opl_status(
        self,
        opl_id: str,
        *,
        status: str,
        apply_status: str,
        apply_error: str = "",
        applied_at: str = "",
        applied_entity_sha256: str = "",
    ) -> dict[str, object]:
        key = self._find_op_key("OPL", opl_id)
        if not key:
            raise FileNotFoundError(f"OPL not found: {opl_id}")
        content = self.store.read_text(key) or ""
        if not content:
            raise FileNotFoundError(f"OPL not found: {opl_id}")
        frontmatter = parse_frontmatter(content)
        model = OPLModel(
            opl_id=opl_id,
            title=str(frontmatter.get("title", opl_id)),
            parent_opa=str(frontmatter.get("parent_opa", "")),
            ops_uris=self._list_value(frontmatter.get("ops_uris")),
            target_uri=str(frontmatter.get("target_uri", "")),
            status=status,
            proposal=str(frontmatter.get("proposal", "")),
            rationale=str(frontmatter.get("rationale", "")),
            evidence_uris=self._list_value(frontmatter.get("evidence_uris")),
            related_uris=self._list_value(frontmatter.get("related_uris")),
            source_type=str(frontmatter.get("source_type", "pipeline")),
            expert_id=str(frontmatter.get("expert_id", "")),
            expert_name=str(frontmatter.get("expert_name", "")),
            source_uri=str(frontmatter.get("source_uri", "")),
            target_concept=str(frontmatter.get("target_concept", "")),
            target_class_name=str(frontmatter.get("target_class_name", "")),
            target_object_name=str(frontmatter.get("target_object_name", "")),
            expected_sha256=str(frontmatter.get("expected_sha256", "")),
            candidate_content=str(frontmatter.get("candidate_content", "")),
            candidate_operations=self._list_of_dicts(frontmatter.get("candidate_operations")),
            apply_status=apply_status,
            apply_error=apply_error,
            applied_at=applied_at,
            applied_entity_sha256=applied_entity_sha256,
        )
        return self._write_opl_model(model)

    def _resolve_external_target_conflicts(
        self, target_uri: str, evidence_uris: list[str], opl_uri: str
    ) -> int:
        resolved = 0
        for record in self.get_opas(
            target_uri=target_uri, status="pending", category="conflict", limit=200
        ):
            self.resolve_opa(
                str(record["opa_id"]),
                solution=f"Resolved by external expert OPL {opl_uri}.",
                evidence_uris=evidence_uris,
            )
            resolved += 1
        return resolved

    def apply_opl(self, opl_id: str) -> dict[str, object]:
        """Apply a stored external OPL exactly once when it is machine-ready."""
        key = self._find_op_key("OPL", opl_id)
        if not key:
            raise FileNotFoundError(f"OPL not found: {opl_id}")
        content = self.store.read_text(key) or ""
        if not content:
            raise FileNotFoundError(f"OPL not found: {opl_id}")
        frontmatter = parse_frontmatter(content)
        if str(frontmatter.get("source_type", "")) != "external_expert":
            raise ValueError("Only external_expert OPL records can be auto-applied")
        if str(frontmatter.get("apply_status", "")) == "applied":
            return {
                "opl_id": opl_id,
                "uri": self._op_uri("OPL", opl_id),
                "status": str(frontmatter.get("status", "applied")),
                "apply_status": "applied",
                "idempotent": True,
            }
        target_uri = str(frontmatter.get("target_uri", ""))
        target_info = self.store.lookup_by_uri(target_uri)
        concept = str(frontmatter.get("target_concept", ""))
        class_name = str(frontmatter.get("target_class_name", ""))
        object_name = str(frontmatter.get("target_object_name", ""))
        if target_info is not None:
            concept, class_name, object_name = target_info[0], target_info[1] or "", target_info[2]
        candidate = str(frontmatter.get("candidate_content", ""))
        operations: list[dict[str, object]] = self._list_of_dicts(
            frontmatter.get("candidate_operations")
        )
        if not candidate and not operations:
            return self._update_external_opl_status(
                opl_id, status="unconfirmed", apply_status="needs_review"
            )
        current = self.store.read_entity(concept, class_name or None, object_name)
        expected_sha256 = str(frontmatter.get("expected_sha256", ""))
        if current is not None:
            current_sha256 = sha256(current.encode("utf-8")).hexdigest()
            if not expected_sha256:
                return self._update_external_opl_status(
                    opl_id,
                    status="unconfirmed",
                    apply_status="needs_review",
                    apply_error=(
                        "Existing target requires expected_sha256 before automatic application."
                    ),
                )
            if expected_sha256 != current_sha256:
                return self._update_external_opl_status(
                    opl_id,
                    status="unconfirmed",
                    apply_status="failed",
                    apply_error=(
                        "Target changed after external OPL was prepared; "
                        "re-submit with a fresh expected_sha256."
                    ),
                )
            expected_sha256 = current_sha256
            if not candidate:
                candidate = self._apply_operations(current, operations)
        elif not candidate:
            return self._update_external_opl_status(
                opl_id,
                status="unconfirmed",
                apply_status="failed",
                apply_error=(
                    "candidate_operations cannot create a missing target without candidate_content."
                ),
            )
        try:
            entity_uri = self.merge_entity(
                concept,
                class_name,
                object_name,
                candidate,
                expected_sha256=expected_sha256,
                conflict_policy="external_authority",
            )
        except (FileNotFoundError, OSError, ValueError) as error:
            return self._update_external_opl_status(
                opl_id,
                status="unconfirmed",
                apply_status="failed",
                apply_error=str(error),
            )
        opl_uri = self._op_uri("OPL", opl_id)
        resolved_conflicts = self._resolve_external_target_conflicts(
            target_uri=entity_uri,
            evidence_uris=self._list_value(frontmatter.get("evidence_uris")),
            opl_uri=opl_uri,
        )
        result = self._update_external_opl_status(
            opl_id,
            status="applied",
            apply_status="applied",
            applied_at=datetime.now(UTC).isoformat(),
            applied_entity_sha256=sha256(
                (self.store.read_entity_by_uri(entity_uri) or "").encode("utf-8")
            ).hexdigest(),
        )
        result.update({"entity_uri": entity_uri, "resolved_conflict_count": resolved_conflicts})
        return result

    def get_opls(
        self,
        *,
        parent_opa: str = "",
        status: str = "",
        target_uri: str = "",
        limit: int = 50,
    ) -> list[dict[str, object]]:
        """Read unconfirmed or historical OPL proposals."""
        parent_id = self._record_id(parent_opa, "opa") if parent_opa else ""
        parent_uri = ""
        if parent_id:
            try:
                _pid, _pfm, parent_uri = self._opa_record(parent_id)
            except FileNotFoundError:
                parent_uri = self._opa_uri(parent_id)
        out: list[dict[str, object]] = []
        for path, content in self._op_files("OPL", target_uri=target_uri):
            fm = self._record_from_content(content)
            sections = extract_sections(content)
            if target_uri and str(fm.get("target_uri", "")) != target_uri:
                continue
            if parent_uri and str(fm.get("parent_opa", "")) not in {parent_uri, parent_id}:
                continue
            if status and str(fm.get("status", "")) != status:
                continue
            opl_id = str(fm.get("id", Path(path).stem))
            out.append(
                {
                    "opl_id": opl_id,
                    "uri": self._op_uri("OPL", opl_id),
                    "path": path,
                    "title": str(fm.get("title", opl_id)),
                    "parent_opa": str(fm.get("parent_opa", "")),
                    "ops_uris": self._list_value(fm.get("ops_uris")),
                    "target_uri": str(fm.get("target_uri", "")),
                    "status": str(fm.get("status", "unconfirmed")),
                    "proposal": str(fm.get("proposal", ""))
                    or sections.get("初版知识提案", "").strip(),
                    "rationale": str(fm.get("rationale", ""))
                    or sections.get("形成依据", "").strip(),
                    "evidence_uris": self._list_value(fm.get("evidence_uris")),
                    "related_uris": self._list_value(fm.get("related_uris")),
                    "source_type": str(fm.get("source_type", "pipeline")),
                    "expert_id": str(fm.get("expert_id", "")),
                    "expert_name": str(fm.get("expert_name", "")),
                    "source_uri": str(fm.get("source_uri", "")),
                    "apply_status": str(fm.get("apply_status", "not_applied")),
                    "apply_error": str(fm.get("apply_error", "")),
                    "applied_at": str(fm.get("applied_at", "")),
                },
            )
            if len(out) >= max(1, limit):
                break
        return out

    def op_flow_status(self, *, limit: int = 500) -> dict[str, object]:
        """Check that every still-open OPA has a valid OPS draft or a reasoned deferral.

        "Open" is decided by ``_opa_open_for_gate`` (closure_status, not
        status), so a record marked ``status=resolved`` no longer bypasses
        the gate unless its ``closure_status`` is ``closed`` or
        ``deferred`` (the latter only with a ``closure_reason``).
        """
        checkpoint = self.store.read_json("index/build_checkpoint.json")
        build_id = (
            str(checkpoint.get("build_id", "")).strip() if isinstance(checkpoint, dict) else ""
        )
        all_opas = self.get_opas(build_id=build_id, limit=limit)
        pending = [opa for opa in all_opas if self._opa_open_for_gate(opa)]
        # Load OPS once.  The previous implementation rescanned every OPS
        # file once per pending OPA, turning a quality gate into OPAxOPS remote
        # directory reads on large builds.
        all_ops = self.get_ops(limit=max(10000, limit * 4))
        ops_by_parent: dict[str, list[dict[str, object]]] = {}
        for row in all_ops:
            parent = self._record_id(str(row.get("parent_opa", "")), "opa")
            ops_by_parent.setdefault(parent, []).append(row)
        blockers: list[dict[str, object]] = []
        for opa in pending:
            opa_id = str(opa["opa_id"])
            target_uri = str(opa.get("target_uri", ""))
            target_readable = bool(target_uri and self.read_resource(target_uri) is not None)
            ops = ops_by_parent.get(opa_id, [])
            invalid: list[str] = self._op_reference_quality(
                target_uri=target_uri,
                evidence_uris=self._list_value(opa.get("evidence_uris")),
                related_uris=self._list_value(opa.get("related_uris")),
            )
            required_opa_fields = ("description", "finding", "missing", "recommendation")
            if any(not str(opa.get(field, "")).strip() for field in required_opa_fields):
                invalid.append("OPA_CONTENT_MISSING")
            valid_ops = [
                row
                for row in ops
                if (
                    str(row.get("status", "")) in {"unconfirmed", "confirmed"}
                    and self.store.resolve_redirect(str(row.get("target_uri", "")))
                    == self.store.resolve_redirect(target_uri)
                    and (
                        str(row.get("source_type", "pipeline")) == "external_expert"
                        or (
                            bool(str(row.get("retrieval_query", "")).strip())
                            and self._retrieval_scopes_match(
                                self._list_value(row.get("retrieval_scopes"))
                            )
                        )
                    )
                    and (bool(self._list_value(row.get("evidence_uris"))) or target_readable)
                    and not self._op_reference_quality(
                        target_uri=str(row.get("target_uri", "")),
                        evidence_uris=self._list_value(row.get("evidence_uris")),
                        related_uris=self._list_value(row.get("related_uris")),
                    )
                )
            ]
            if ops and any(
                str(row.get("source_type", "pipeline")) != "external_expert"
                and not str(row.get("retrieval_query", "")).strip()
                for row in ops
            ):
                invalid.append("OPS_RETRIEVAL_RECEIPT_MISSING")
            if ops and any(
                self.store.resolve_redirect(str(row.get("target_uri", "")))
                != self.store.resolve_redirect(target_uri)
                for row in ops
            ):
                invalid.append("OPS_TARGET_MISMATCH")
            if ops and any(
                self._list_value(row.get("retrieval_used_uris"))
                and (
                    not set(self._list_value(row.get("retrieval_used_uris"))).issubset(
                        set(self._list_value(row.get("retrieval_hit_uris"))),
                    )
                    or not set(self._list_value(row.get("retrieval_used_uris"))).issubset(
                        set(self._list_value(row.get("evidence_uris")))
                        | set(self._list_value(row.get("related_uris")))
                        | {str(row.get("target_uri", ""))},
                    )
                )
                for row in ops
            ):
                invalid.append("OPS_RETRIEVAL_USAGE_MISMATCH")
            if not valid_ops or invalid:
                blockers.append(
                    {
                        "opa_id": opa_id,
                        "opa_uri": opa["uri"],
                        "status": str(opa.get("status", "")),
                        "closure_status": str(opa.get("closure_status", "")),
                        "missing": ["OPS"] if not valid_ops else [],
                        "invalid": invalid,
                    },
                )
        return {
            "passed": not blockers,
            "open_opa_count": len(pending),
            "blocker_count": len(blockers),
            "blockers": blockers,
        }

    def ops_dispatch_plan(
        self,
        *,
        limit: int = 1000,
        max_parallel_shards: int | None = None,
    ) -> dict[str, object]:
        """Code-generated, domain-grouped OPS dispatch plan for the conductor.

        Returns every open OPA that has no OPS draft yet, sorted cheapest
        repair first (``relation_missed`` before ``source_incomplete``) and
        grouped by target class so the conductor can dispatch the whole plan
        in one ``task_create_batch`` instead of hand-enumerating.

        The plan also exposes a sharded view (``shards``): items grouped by
        target class and chunked into ``_entity_batch_limit()``-sized OPS
        work batches. ``max_parallel_shards`` caps how many shards are
        dispatched in the current wave; ``remaining_count`` reports how many
        shards are left for later waves.
        """
        if max_parallel_shards is not None and max_parallel_shards < 1:
            raise ValueError("max_parallel_shards must be positive when provided")
        checkpoint = self.store.read_json("index/build_checkpoint.json")
        build_id = (
            str(checkpoint.get("build_id", "")).strip() if isinstance(checkpoint, dict) else ""
        )
        all_opas = self.get_opas(build_id=build_id, limit=limit)
        open_opas = [opa for opa in all_opas if self._opa_open_for_gate(opa)]
        all_ops = self.get_ops(limit=max(10000, limit * 4))
        ops_parent_ids = {self._record_id(str(row.get("parent_opa", "")), "opa") for row in all_ops}
        # Cheapest repair first so the conductor can close link gaps quickly.
        repair_cost = {
            "relation_missed": 0,
            "extraction_missed": 1,
            "param_unhosted": 2,
            "source_incomplete": 3,
            "manual_error": 4,
            "process_conflict": 5,
            "hallucination": 6,
            "content_missing": 7,
            "fact_conflict": 8,
        }
        root_prefix = self.store.root_uri.rstrip("/") + "/"
        items: list[dict[str, object]] = []
        for opa in open_opas:
            opa_id = str(opa["opa_id"])
            if opa_id in ops_parent_ids:
                continue
            target_uri = str(opa.get("target_uri", ""))
            target_class = target_uri.removeprefix(root_prefix).split("/")[0] if target_uri else ""
            reason = str(opa.get("reason_code", "content_missing"))
            section = str(opa.get("target_section", ""))
            items.append(
                {
                    "opa_id": opa_id,
                    "opa_uri": str(opa.get("uri", "")),
                    "target_uri": target_uri,
                    "target_class": target_class,
                    "target_section": section,
                    "reason_code": reason,
                    "suggested_retrieval_query": f"{opa.get('title', '')!s} {section}".strip(),
                    "repair_cost": repair_cost.get(reason, 9),
                },
            )
        items.sort(key=lambda row: (row["repair_cost"], row["target_class"], row["opa_id"]))
        by_domain: dict[str, int] = {}
        for row in items:
            by_domain[str(row["target_class"])] = by_domain.get(str(row["target_class"]), 0) + 1
        grouped_by_class: dict[str, list[dict[str, object]]] = {}
        for row in items:
            grouped_by_class.setdefault(str(row["target_class"]), []).append(row)
        all_shards: list[dict[str, object]] = []
        shard_index = 0
        for target_class in sorted(grouped_by_class):
            class_items = grouped_by_class[target_class]
            for start in range(0, len(class_items), _entity_batch_limit()):
                shard_index += 1
                chunk = class_items[start : start + _entity_batch_limit()]
                opa_ids = [str(row["opa_id"]) for row in chunk]
                shard_id = f"ops_{target_class.casefold()}_{shard_index}"
                task_description = "\n".join(
                    [
                        "worker_role: wiki_ops_worker",
                        "depends_on_stage: opa_discovered",
                        "ops_planner: ops_dispatch_plan",
                        f"ops_shard_id: {shard_id}",
                        f"write_scope: ops_draft:{shard_id}",
                        f"opa_ids: {json.dumps(opa_ids, ensure_ascii=False)}",
                        (
                            "worker_task: read each OPA and its evidence, then create an "
                            "unconfirmed OPS draft proposing the repair for that OPA"
                        ),
                        "expected_artifacts: unconfirmed_ops_drafts",
                    ],
                )
                all_shards.append(
                    {
                        "shard_id": shard_id,
                        "worker_role": "wiki_ops_worker",
                        "opa_ids": opa_ids,
                        "item_count": len(chunk),
                        "task_description": task_description,
                    },
                )
        shards = all_shards[:max_parallel_shards] if max_parallel_shards is not None else all_shards
        remaining_count = len(all_shards) - len(shards)
        return {
            "build_id": build_id,
            "open_opa_count": len(open_opas),
            "to_dispatch": len(items),
            "already_has_ops": len(open_opas) - len(items),
            "by_domain": by_domain,
            "items": items,  # keep flat list for backward compat
            "shards": shards,  # NEW: sharded view
            "remaining_count": remaining_count,  # NEW
        }

    def op_flow_report(
        self,
        *,
        persist: bool = True,
        limit: int = 10000,
        build_id: str | None = None,
    ) -> dict[str, object]:
        """Report OPA/OPS/OPL counts and source/Wiki evidence coverage.

        Coverage is calculated from readable raw or Wiki URIs, not merely from
        a non-empty YAML field.  OPL counts are observational only; the
        automatic gate is based on valid OPS drafts. The report is persisted under
        ``index/op_flow_report.json`` by default for build handoff.
        """
        if limit < 1 or limit > _MAX_REPORT_LIMIT:
            raise ValueError("op_flow_report limit must be between 1 and 10000")
        opas = self.get_opas(limit=limit, build_id=build_id or "")
        ops = self.get_ops(limit=limit)
        opls = self.get_opls(limit=limit)
        if build_id is not None:
            active_opa_ids = {str(row.get("opa_id", "")) for row in opas}
            ops = [
                row
                for row in ops
                if self._record_id(str(row.get("parent_opa", "")), "opa") in active_opa_ids
            ]
            opls = [
                row
                for row in opls
                if self._record_id(str(row.get("parent_opa", "")), "opa") in active_opa_ids
            ]

        def coverage(uris: list[str]) -> dict[str, object]:
            unique = list(dict[str, object].fromkeys(uri.strip() for uri in uris if uri.strip()))
            manual = 0
            wiki = 0
            readable = 0
            unresolved = 0
            for uri in unique:
                kind = classify_raw_source_uri(uri, raw_root_uri=self._raw_fs.root_uri)
                if kind is not None:
                    if kind.value == "manual_chapter":
                        manual += 1
                    if self.read_raw_source(uri).status is SourceReadStatus.OK:
                        readable += 1
                    else:
                        unresolved += 1
                    continue
                if self.store.is_wiki_uri(uri):
                    wiki += 1
                    if self.read_resource(uri) is not None:
                        readable += 1
                    else:
                        unresolved += 1
                    continue
                unresolved += 1
            return {
                "reference_count": len(unique),
                "readable_source_or_wiki_count": readable,
                "manual_source_count": manual,
                "wiki_reference_count": wiki,
                "unresolved_reference_count": unresolved,
                "coverage_percent": round((readable / len(unique)) * 100, 2) if unique else 0.0,
            }

        opa_rows: list[dict[str, object]] = []
        for record in opas:
            refs = [
                str(record.get("target_uri", "")),
                *self._list_value(record.get("evidence_uris")),
                *self._list_value(record.get("related_uris")),
            ]
            opa_rows.append(
                {
                    "id": record.get("opa_id", ""),
                    "status": record.get("status", ""),
                    "category": record.get("category", ""),
                    "target_uri": record.get("target_uri", ""),
                    "coverage": coverage(refs),
                },
            )
        ops_rows: list[dict[str, object]] = []
        for record in ops:
            refs = [
                str(record.get("target_uri", "")),
                *self._list_value(record.get("evidence_uris")),
                *self._list_value(record.get("related_uris")),
                *self._list_value(record.get("retrieval_used_uris")),
            ]
            ops_rows.append(
                {
                    "id": record.get("ops_id", ""),
                    "status": record.get("status", ""),
                    "parent_opa": record.get("parent_opa", ""),
                    "coverage": coverage(refs),
                },
            )
        opl_rows: list[dict[str, object]] = []
        for record in opls:
            refs = [
                str(record.get("target_uri", "")),
                *self._list_value(record.get("evidence_uris")),
                *self._list_value(record.get("ops_uris")),
            ]
            opl_rows.append(
                {
                    "id": record.get("opl_id", ""),
                    "status": record.get("status", ""),
                    "parent_opa": record.get("parent_opa", ""),
                    "coverage": coverage(refs),
                },
            )

        def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
            covered = 0
            for row in rows:
                coverage = row.get("coverage")
                if isinstance(coverage, dict):
                    count = coverage.get("readable_source_or_wiki_count", 0)
                    if isinstance(count, (int, float)) and count > 0:
                        covered += 1
            return {
                "total": len(rows),
                "with_readable_source_or_wiki": covered,
                "coverage_percent": round((covered / len(rows)) * 100, 2) if rows else 0.0,
            }

        report: dict[str, object] = {
            "version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "counts": {
                "opa": len(opas),
                "ops": len(ops),
                "opl": len(opls),
                "opa_pending": sum(1 for row in opas if row.get("status") == "pending"),
                "ops_unconfirmed": sum(1 for row in ops if row.get("status") == "unconfirmed"),
                "ops_confirmed": sum(1 for row in ops if row.get("status") == "confirmed"),
                "ops_rejected": sum(1 for row in ops if row.get("status") == "rejected"),
                "ops_applied": sum(1 for row in ops if row.get("apply_status") == "applied"),
                "opl_unconfirmed": sum(1 for row in opls if row.get("status") == "unconfirmed"),
            },
            "coverage": {
                "opa": aggregate(opa_rows),
                "ops": aggregate(ops_rows),
                "opl": aggregate(opl_rows),
            },
            "records": {"opa": opa_rows, "ops": ops_rows, "opl": opl_rows},
        }
        if persist:
            self.store.write_json("index/op_flow_report.json", report)
        return report

    def discover_opa(
        self,
        *,
        profile: BuildProfile = "manual",
        entity_uris: list[str] | None = None,
        include_warnings: bool = False,
        limit: int = 500,
    ) -> dict[str, object]:
        """Materialize stable pending OPAs from the complete audit.

        This is mechanical: audit remains the source of truth, while workers
        may enrich or resolve the OPA later. Repeated discovery reuses
        ``create_opa``'s stable identity and never bypasses finalize.
        """
        if limit < 1 or limit > _MAX_DISCOVERY_LIMIT:
            raise ValueError("discover_opa limit must be between 1 and 500")
        issues: list[dict[str, object]] = []
        offset = 0
        while True:
            report = self.audit_wiki(
                profile=profile,
                entity_uris=entity_uris,
                offset=offset,
                limit=limit,
            )
            report_issues = report.get("issues", [])
            if isinstance(report_issues, list):
                issues.extend(dict(issue) for issue in report_issues if isinstance(issue, dict))
            next_offset_raw = report.get("next_offset", -1)
            next_offset = (
                int(next_offset_raw) if isinstance(next_offset_raw, (int, float, str)) else -1
            )
            if next_offset < 0:
                break
            offset = next_offset

        # Discovery is append/reuse-only for the current build.  It must not
        # mutate or reconcile OPA records created by an earlier build.

        created: list[dict[str, object]] = []
        skipped: list[dict[str, str]] = []
        unclassified: list[dict[str, str]] = []
        repair_only_count = 0
        unclassified_count = 0
        skipped_low_value = 0
        # Cross-issue caches: avoid re-reading the same entity / raw source
        # for every issue that targets the same URI.  Without these caches
        # discover_opa makes ~1500+ HTTP round-trips for a 500-issue audit.
        entity_cache: dict[str, str | None] = {}
        raw_cache: dict[str, SourceReadResult] = {}

        def _cached_read_entity(uri: str) -> str | None:
            if uri not in entity_cache:
                entity_cache[uri] = self.read_resource(uri)
            return entity_cache[uri]

        def _cached_read_raw(uri: str) -> SourceReadResult:
            if uri not in raw_cache:
                raw_cache[uri] = self.read_raw_source(uri)
            return raw_cache[uri]

        for issue in issues:
            severity = str(issue.get("severity", "error"))
            if severity != "error" and not include_warnings:
                continue
            target_uri = str(issue.get("uri", "")).strip()
            if not self.store.is_wiki_uri(target_uri) or target_uri == self.store.root_uri + "/":
                skipped.append(
                    {
                        "code": str(issue.get("code", "")),
                        "reason": "audit issue has no concrete wiki target",
                    },
                )
                continue
            code = str(issue.get("code", "audit_issue")).strip() or "audit_issue"
            concept = str(issue.get("concept", "")).strip()
            if is_optional_relation_issue(code, concept):
                skipped.append({"code": code, "reason": "optional relation under backbone policy"})
                continue
            if _is_relation_gap_code(code):
                skipped.append({"code": code, "reason": "relation gap excluded from OPA"})
                continue
            message = str(issue.get("message", "")).strip() or code
            disposition_value = str(issue.get("disposition", "")).strip()
            reason_code = str(issue.get("opa_reason_code", "")).strip()
            if not disposition_value:
                policy = audit_issue_policy(code)
                if policy is not None:
                    disposition_value = policy.disposition.value
                    reason_code = reason_code or policy.opa_reason_code
            if disposition_value == IssueDisposition.REPAIR_ONLY.value:
                # repair_only means the target concept is already materialized
                # but a content/link gap remains. Convert to an open_gap OPA
                # instead of silently skipping, so the finalize gate can see
                # the tracked record and the limitation is explicitly recorded.
                if not reason_code:
                    reason_code = "content_missing"
                disposition_value = IssueDisposition.GAP.value
            if (
                disposition_value
                not in {
                    IssueDisposition.GAP.value,
                    IssueDisposition.CONFLICT.value,
                }
                or not reason_code
            ):
                unclassified_count += 1
                item = {"code": code, "reason": "audit issue has no OPA disposition"}
                skipped.append(item)
                unclassified.append(item)
                continue
            # Skip known low-value issue classes before materializing an OPA.
            # The audit prompt already avoids these; enforce it in code so a
            # prompt drift cannot flood the OPA ledger with noise.
            target_section = self._opa_target_section(message, code)
            root_prefix = self.store.root_uri.rstrip("/") + "/"
            target_class = target_uri.removeprefix(root_prefix).split("/")[0] if target_uri else ""
            is_low_value = (
                reason_code == "relation_missed"
                or (
                    target_section.startswith("frontmatter:")
                    and reason_code not in {"content_missing", "fact_conflict"}
                )
                or (
                    target_class in {"Procedure", "DTC", "Part"}
                    and reason_code in {"relation_missed", "extraction_missed"}
                )
            )
            if is_low_value:
                skipped_low_value += 1
                skipped.append({"code": code, "reason": "low-value audit issue"})
                continue
            category = disposition_value
            content = _cached_read_entity(target_uri) or ""
            frontmatter = parse_frontmatter(content)
            evidence = self._list_value(frontmatter.get("sources"))
            evidence.extend(extract_source_uris(message))
            evidence = list(
                dict[str, object].fromkeys(
                    uri for uri in evidence if _cached_read_raw(uri).status is SourceReadStatus.OK
                )
            )
            # A missing/invalid source citation is itself an audit finding.  Do
            # not drop that finding merely because the page has no usable raw
            # URI: bind the OPA to the concrete page as observation evidence,
            # while keeping ``missing`` explicit so this is never mistaken for
            # a source citation.
            if not evidence:
                unclassified_count += 1
                item = {"code": code, "reason": "OPA issue has no readable evidence URI"}
                skipped.append(item)
                unclassified.append(item)
                continue
            related = list(
                dict[str, object].fromkeys(
                    uri
                    for uri in re.findall(re.escape(self.store.root_uri) + r"/[^\s)\]>]+", message)
                    if uri != target_uri
                )
            )
            category_label = "内容缺失" if category == "gap" else "知识冲突"
            issue_concept = str(issue.get("concept", "")).strip()
            entity_name = target_uri.rstrip("/").rsplit("/", 1)[-1].removesuffix(".md")
            if entity_name and entity_name not in issue_concept:
                prefix = f"{category_label}: {entity_name}"
            else:
                prefix = f"{category_label}:"
            title = f"{prefix} {issue_concept} {target_section}".strip()
            result = self.create_opa(
                title=title,
                description=message,
                category=category,
                reason_code=reason_code,
                target_uri=target_uri,
                target_section=target_section,
                source_chapter=evidence[0] if evidence else "",
                evidence_uris=evidence,
                finding=message,
                missing=self._opa_missing_text(code, message),
                recommendation=self._opa_recommendation(reason_code),
                related_uris=related,
            )
            created.append(result)
        return {
            "issues_scanned": len(issues),
            "opa_count": len(created),
            "reconciled_count": 0,
            "profile": profile,
            "repair_only_count": repair_only_count,
            "unclassified_count": unclassified_count,
            "skipped_low_value": skipped_low_value,
            "created": created,
            "skipped": skipped,
            "unclassified": unclassified,
        }

    @staticmethod
    def _opa_target_section(message: str, code: str) -> str:
        """Extract a stable section label from an audit issue."""
        match = re.search(r"section\s+['\"]([^'\"]+)['\"]", message, re.IGNORECASE)
        return match.group(1).strip() if match is not None else code

    @staticmethod
    def _opa_missing_text(code: str, message: str) -> str:
        """Describe the unresolved fact without inventing a domain answer."""
        return message

    @staticmethod
    def _opa_recommendation(reason_code: str) -> str:
        """Provide a deterministic next action for the worker queue."""
        if reason_code == "fact_conflict":
            return "保留冲突双方及证据,完成裁决前不得静默覆盖或移动实体。"
        return "回读实体与 raw 来源补齐内容;原文确实缺失时保留 open_gap。"

    def resolve_opa(
        self,
        opa_id: str,
        *,
        solution: str,
        evidence_uris: list[str] | None = None,
        related_uris: list[str] | None = None,
        closure_status: str = "deferred",
        closure_reason: str = "",
    ) -> dict[str, object]:
        """Resolve an existing OPA while retaining its complete evidence chain.

        ``closure_status`` defaults to ``deferred``: resolving a record
        without applying a fix is an honest deferral, not a closure. Callers
        that actually applied the fix (e.g. ``apply_opa``) pass
        ``closure_status="closed"``.
        """
        key = self._find_opa_key(opa_id)
        if key is None:
            raise FileNotFoundError(f"OPA not found: {opa_id}")
        content = self.store.read_text(key)
        if content is None:
            raise FileNotFoundError(f"OPA not found: {opa_id}")
        frontmatter = self._record_from_content(content)
        return self.create_opa(
            opa_id=opa_id,
            title=str(frontmatter.get("title", opa_id)),
            description=str(frontmatter.get("description", "")),
            category=str(frontmatter.get("category", "conflict")),
            reason_code=str(
                frontmatter.get(
                    "reason_code",
                    infer_opa_reason_code(str(frontmatter.get("category", "conflict"))),
                ),
            ),
            scope=str(frontmatter.get("scope", "entity")),
            subtype=str(frontmatter.get("subtype", "wiki_error")),
            target_uri=str(frontmatter.get("target_uri", "")),
            target_path=str(frontmatter.get("target_path", "")),
            target_section=str(frontmatter.get("target_section", "")),
            source_chapter=str(frontmatter.get("source_chapter", "")),
            evidence_uris=evidence_uris,
            status="resolved",
            solution=solution,
            finding=str(frontmatter.get("finding", "")),
            missing=str(frontmatter.get("missing", "")),
            recommendation=str(frontmatter.get("recommendation", "")),
            related_uris=related_uris,
            build_id=str(frontmatter.get("build_id", "")).strip(),
            closure_status=closure_status,
            closure_reason=closure_reason or str(frontmatter.get("closure_reason", "")),
        )

    def apply_opa(
        self,
        opa_id: str,
        *,
        concept: str,
        class_name: str,
        object_name: str,
        content: str,
        solution: str,
        expected_sha256: str = "",
    ) -> dict[str, object]:
        """Apply a resolved candidate and close its OPA in one guarded call."""
        key = self._find_opa_key(opa_id)
        if key is None:
            raise FileNotFoundError(f"OPA not found: {opa_id}")
        stored = self.store.read_text(key)
        if stored is None:
            raise FileNotFoundError(f"OPA not found: {opa_id}")
        frontmatter = self._record_from_content(stored)
        target_uri = str(frontmatter.get("target_uri", ""))
        expected_uri = self.store.entity_uri(concept, class_name or None, object_name)
        if target_uri and self.store.resolve_redirect(target_uri) != self.store.resolve_redirect(
            expected_uri
        ):
            raise ValueError(
                f"OPA target does not match merge target: {target_uri} != {expected_uri}",
            )
        merge = self.merge_entity(
            concept,
            class_name,
            object_name,
            stored,
            expected_sha256=expected_sha256,
        )
        resolved = self.resolve_opa(
            opa_id,
            solution=solution,
            evidence_uris=[expected_uri],
            closure_status="closed",
            closure_reason="applied via merge_entity",
        )
        return {"entity_uri": merge, "opa": resolved}

    def refine_opa_reason_code(
        self,
        opa_id: str,
        *,
        reason_code: str,
        closure_status: str = "",
        closure_reason: str = "",
    ) -> dict[str, object]:
        """Refine an OPA's reason_code after retrieval-based triage (called by OPS).

        OPA are created with a coarse reason (``content_missing`` /
        ``fact_conflict``) because at audit time the source has not yet been
        retrieved. After OPS runs retrieval it can classify the gap precisely
        into one of the fine codes (``source_incomplete`` / ``extraction_missed``
        / ``relation_missed`` / ``param_unhosted`` / ``manual_error`` /
        ``process_conflict`` / ``hallucination``), each of which maps to a
        distinct repair path. This rewrites the OPA in place preserving all
        prior evidence.
        """
        if reason_code not in OPA_REASON_CODES:
            raise ValueError(
                f"Unsupported reason_code: {reason_code!r}; "
                f"expected one of {', '.join(OPA_REASON_CODES)}",
            )
        key = self._find_opa_key(opa_id)
        if key is None:
            raise FileNotFoundError(f"OPA not found: {opa_id}")
        content = self.store.read_text(key)
        if content is None:
            raise FileNotFoundError(f"OPA not found: {opa_id}")
        frontmatter = self._record_from_content(content)
        effective_closure = closure_status or str(frontmatter.get("closure_status", ""))
        if effective_closure and effective_closure not in OPA_CLOSURE_STATUSES:
            raise ValueError(f"Unsupported closure_status: {effective_closure!r}")
        return self.create_opa(
            opa_id=opa_id,
            title=str(frontmatter.get("title", opa_id)),
            description=str(frontmatter.get("description", "")),
            category=str(frontmatter.get("category", "conflict")),
            reason_code=reason_code,
            scope=str(frontmatter.get("scope", "entity")),
            subtype=str(frontmatter.get("subtype", "wiki_error")),
            target_uri=str(frontmatter.get("target_uri", "")),
            target_path=str(frontmatter.get("target_path", "")),
            target_section=str(frontmatter.get("target_section", "")),
            source_chapter=str(frontmatter.get("source_chapter", "")),
            evidence_uris=self._list_value(frontmatter.get("evidence_uris")),
            status=str(frontmatter.get("status", "pending")),
            solution=str(frontmatter.get("solution", "")),
            finding=str(frontmatter.get("finding", "")),
            missing=str(frontmatter.get("missing", "")),
            recommendation=str(frontmatter.get("recommendation", "")),
            related_uris=self._list_value(frontmatter.get("related_uris")),
            build_id=str(frontmatter.get("build_id", "")).strip(),
            closure_status=effective_closure
            or str(frontmatter.get("closure_status", "open"))
            or "open",
            closure_reason=closure_reason or str(frontmatter.get("closure_reason", "")),
        )

    def get_opas(
        self,
        *,
        target_uri: str = "",
        status: str = "",
        category: str = "",
        reason_code: str = "",
        scope: str = "",
        source_chapter: str = "",
        limit: int = 50,
        include_superseded: bool = False,
        build_id: str = "",
    ) -> list[dict[str, object]]:
        """读取 ``OP/OpA/`` 下匹配条件的 OPA 记录。.

        可按 ``target_uri`` / ``status`` / ``category`` / ``reason_code`` / ``scope`` 过滤,
        返回结构化 frontmatter(含 ``uri`` 键,供证据链提取)。
        """
        out: list[dict[str, object]] = []
        for path, content in self._opa_files(target_uri=target_uri):
            fm = self._record_from_content(content)
            opa_id = str(fm.get("id", Path(path).stem))
            if not include_superseded and str(fm.get("status", "")).strip().lower() in {
                "superseded",
                "rejected",
            }:
                continue
            if target_uri and str(fm.get("target_uri", "")) != target_uri:
                continue
            if status and str(fm.get("status", "")) != status:
                continue
            if category and str(fm.get("category", "")) != category:
                continue
            effective_reason_code = str(
                fm.get("reason_code", infer_opa_reason_code(str(fm.get("category", "")))),
            )
            if reason_code and effective_reason_code != reason_code:
                continue
            if scope and str(fm.get("scope", "")) != scope:
                continue
            if source_chapter and str(fm.get("source_chapter", "")) != source_chapter:
                continue
            if build_id and not self._opa_matches_build(path, fm, build_id):
                continue
            out.append(
                {
                    "opa_id": opa_id,
                    "path": path,
                    **{
                        key: fm.get(key, "")
                        for key in (
                            "title",
                            "human_key",
                            "description",
                            "category",
                            "reason_code",
                            "scope",
                            "subtype",
                            "target_uri",
                            "target_path",
                            "target_section",
                            "source_chapter",
                            "status",
                            "solution",
                            "finding",
                            "missing",
                            "recommendation",
                            "report_count",
                            "dedupe_key",
                            "build_id",
                            "closure_status",
                            "closure_reason",
                        )
                    },
                    "evidence_uris": self._list_value(fm.get("evidence_uris")),
                    "related_uris": self._list_value(fm.get("related_uris")),
                    "uri": self._opa_uri(opa_id, self._opa_category_from_key(path)),
                },
            )
            if len(out) >= max(limit, 1):
                break
        return out

    def get_expert_authority(
        self, *, target_uri: str = "", limit: int = 50
    ) -> list[dict[str, object]]:
        """Inspect sections protected by confirmed or applied expert knowledge.

        Expert authority comes from confirmed OPS recommendations and
        applied OPL proposals persisted under ``OP/OpA/`` — the same
        records OPA/OPS/OPL tickets are built from.
        """
        authority: list[dict[str, object]] = []
        for path, content in self._op_files("OPL", target_uri=target_uri):
            fm = parse_frontmatter(content)
            if target_uri and str(fm.get("target_uri", "")) != target_uri:
                continue
            if str(fm.get("apply_status", "not_applied")) != "applied":
                continue
            opl_id = str(fm.get("id", Path(path).stem))
            authority.append(
                {
                    "source": "opl",
                    "uri": self._op_uri("OPL", opl_id),
                    "target_uri": str(fm.get("target_uri", "")),
                    "target_section": str(fm.get("target_section", "")),
                    "status": str(fm.get("status", "")),
                    "apply_status": "applied",
                    "applied_at": str(fm.get("applied_at", "")),
                    "expert_id": str(fm.get("expert_id", "")),
                },
            )
            if len(authority) >= max(1, limit):
                return authority
        confirmed = self.get_ops(status="confirmed", target_uri=target_uri, limit=limit)
        authority.extend(
            {
                "source": "ops",
                "uri": str(record.get("uri", "")),
                "target_uri": str(record.get("target_uri", "")),
                "target_section": str(record.get("target_section", "")),
                "status": "confirmed",
                "apply_status": str(record.get("apply_status", "")),
                "applied_at": str(record.get("applied_at", "")),
                "expert_id": str(record.get("expert_id", "")),
            }
            for record in confirmed
        )
        return authority[: max(1, limit)]

    def get_wiki_change_events(
        self, *, target_uri: str = "", limit: int = 50
    ) -> list[dict[str, object]]:
        """Consume durable apply events from applied OPL proposals."""
        events: list[dict[str, object]] = []
        for path, content in self._op_files("OPL", target_uri=target_uri):
            fm = parse_frontmatter(content)
            if target_uri and str(fm.get("target_uri", "")) != target_uri:
                continue
            if str(fm.get("apply_status", "")) != "applied":
                continue
            opl_id = str(fm.get("id", Path(path).stem))
            events.append(
                {
                    "event": "opl_applied",
                    "uri": self._op_uri("OPL", opl_id),
                    "target_uri": str(fm.get("target_uri", "")),
                    "applied_at": str(fm.get("applied_at", "")),
                    "expert_id": str(fm.get("expert_id", "")),
                },
            )
            if len(events) >= max(1, limit):
                break
        return events
