"""Audit mixin for WikiBuildTools — full-library quality audit + paginated reporting."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from typing import TypedDict

from wolfharness.capabilities.wiki.quality import (
    BUILD_PROFILES,
    BuildProfile,
    CoverageSummary,
    IssueDisposition,
    QualityIssue,
    SourceReadResult,
    SourceReadStatus,
    WikiAuditReport,
    audit_issue_policy,
    classify_raw_source_uri,
    confirmation_requirements,
    entity_status,
    extract_malformed_wiki_uris,
    extract_source_uris,
    extract_wiki_uris,
    force_confirmed_status,
    has_unresolved_placeholder,
    parse_frontmatter,
    wiki_uri_prefix,
)
from wolfharness.capabilities.wiki.section_constants import SECTION_COMMON_FAULTS
from wolfharness.capabilities.wiki.tickets.opa import opa_section_names
from wolfharness.capabilities.wiki.validation import ENTITY_VALIDATION_HOOKS, run_entity_validation
from wolfharness.capabilities.wiki.wiki_build_deps import WikiBuildDeps


_CASE_SOFT_REQUIREMENT_CODES = {
    "Component.assembly_parts",
    "Component.procedure.body_link",
    "Component.fault.body_link",
    "DTC.diagnostic_procedure.body_link",
    "Device.diagnostic_chain.body_link",
    "Device.dtc.body_link",
    "Fault.verification_procedures",
    "Fault.repair_procedures",
    "Fault.verification_procedure.body_link",
    "Fault.repair_procedure.body_link",
    "Procedure.target_components",
    "SymptomProfile.diagnostic_procedure.body_link",
}

_GAP_CATEGORY_MAP: dict[str, str] = {
    "Fault.symptom.body_link": "dependency_gap",
    "DTC.diagnostic_procedure.body_link": "dependency_gap",
    "Fault.procedures": "dependency_gap",
    "Fault.verification_procedures": "dependency_gap",
    "Fault.repair_procedures": "dependency_gap",
    "Profile.possible_faults": "dependency_gap",
    "dangling_reference": "dependency_gap",
    "source_unresolvable": "source_unresolvable",
}


_CONTENT_HOOK_NAMES = frozenset({
    "body_sections",
})


def _classify_gap_category(
    code: str,
    opa_reason_code: str = "",
    disposition: str = "",
) -> str:
    """Map an audit issue code to a gap triage category.

    Content-completeness hook findings (``hook.body_sections``, etc.) and
    ``repair_only`` issues on already-materialized entities are content
    limitations, not structural dependency gaps — classify them as
    ``data_limitation`` so the finalize gate does not self-lock on its own
    promotion artifacts.
    """
    if code in _GAP_CATEGORY_MAP:
        return _GAP_CATEGORY_MAP[code]
    if opa_reason_code and opa_reason_code in _GAP_CATEGORY_MAP:
        return _GAP_CATEGORY_MAP[opa_reason_code]
    if code.startswith("hook.") and code.removeprefix("hook.") in _CONTENT_HOOK_NAMES:
        return "data_limitation"
    if disposition == "repair_only":
        return "data_limitation"
    if "物化身份" in code or "model" in code.lower():
        return "data_limitation"
    return "dependency_gap"


_TYPED_RELATION_FIELDS: dict[str, dict[str, str]] = {
    "Device": {"symptom_refs": "Symptom", "critical_components": "Component"},
    "Component": {"assembly_parts": "Part"},
    "Part": {"parent_components": "Component", "procedure_refs": "Procedure"},
    "DTC": {"controller_component": "Component", "related_faults": "Fault"},
    "SymptomProfile": {
        "parent_symptom": "Symptom",
        "device_refs": "Device",
        "direct_component_uri": "Component",
        "possible_faults": "Fault",
    },
    "Fault": {
        "affected_components": "Component",
        "verification_procedures": "Procedure",
        "repair_procedures": "Procedure",
    },
    "Procedure": {"target_components": "Component"},
}

_REPAIR_ACTION_BY_CODE: dict[str, str] = {
    "Fault.affected_components": "在 frontmatter affected_components 中链接受影响的部件 URI。",
    "Fault.procedures": "为该故障构建验证或修复流程，并在 verification_procedures/repair_procedures 中引用。",
    "Fault.verification_procedures": "物化该故障所需的验证流程实体并建立链接。",
    "Fault.repair_procedures": "物化该故障所需的修复/更换流程实体并建立链接。",
    "Fault.verification_procedure.body_link": "将「验证方法」章节链接到已有的验证流程 URI。",
    "Fault.repair_procedure.body_link": "将「修复方式」章节链接到已有的修复/更换流程 URI。",
    "Fault.component.body_link": "将「影响范围」章节链接受影响的部件 URI。",
    "Fault.symptom.body_link": "将「关联故障现象」章节链接到可引发的故障现象 URI。",
    "DTC.controller_role": "在 frontmatter controller_role 中保留源文档确认的控制器功能角色。",
    "DTC.related_faults": "物化或链接该 DTC 引发的故障实体。",
    "DTC.related_faults.body_link": "将「可能失效机理」章节链接到关联的故障 URI。",
    "DTC.diagnostic_procedure.body_link": "物化或链接该 DTC 的诊断流程。",
    "SymptomProfile.device_refs": "在 device_refs 中链接该故障现象适用的具体设备实体。",
    "SymptomProfile.direct_component_uri": "在 direct_component_uri 中链接该故障现象的直接部件 URI。",
    "SymptomProfile.possible_faults": "链接该故障现象可能的故障实体。",
    "SymptomProfile.possible_faults.body_link": "将「可能失效机理」章节链接到可能的故障 URI。",
    "SymptomProfile.diagnostic_procedure.body_link": "将「推荐诊断流程」章节链接到诊断流程 URI。",
    "Device.critical_components": "引用该设备的关重件清单。",
    "Device.symptom_refs": "索引该设备可用的故障现象实体。",
    "Device.diagnostic_chain.body_link": f"在「{SECTION_COMMON_FAULTS}」表中链接故障现象/故障/部件 URI。",
    "Procedure.target_components": "引用该流程的目标部件。",
}


def _repair_action_for(code: str, target_concepts: tuple[str, ...]) -> str:
    """Return a concrete remediation instruction for a REPAIR_ONLY issue."""
    return _REPAIR_ACTION_BY_CODE.get(
        code,
        f"物化或链接缺失的{'、'.join(target_concepts) or '目标'}实体，并在本页引用其 URI。",
    )


def _resource_uris(value: object, *, raw_root_uri: str = "") -> list[str]:
    """Collect resource URIs from YAML lists/maps without using labels.

    Recognises wiki entity URIs (under the active wiki namespace) and
    cross-namespace case/fault-annotation evidence URIs (matching by shape,
    consistent with opa.py and record_source_packet).
    """
    wiki_prefix = wiki_uri_prefix() + "/"
    if isinstance(value, str):
        stripped = value.strip()
        if (
            stripped.startswith(wiki_prefix)
            or classify_raw_source_uri(
                stripped,
                raw_root_uri=raw_root_uri,
            )
            is not None
        ):
            return [stripped]
        return []
    if isinstance(value, dict):
        result: list[str] = []
        for nested in value.values():
            result.extend(_resource_uris(nested, raw_root_uri=raw_root_uri))
        return result
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for nested in value:
            result.extend(_resource_uris(nested, raw_root_uri=raw_root_uri))
        return result
    return []


def _raw_source_uris(value: object, *, raw_root_uri: str) -> list[str]:
    """Collect only provenance URIs from a structured field."""
    return [
        uri
        for uri in _resource_uris(value, raw_root_uri=raw_root_uri)
        if classify_raw_source_uri(uri, raw_root_uri=raw_root_uri) is not None
    ]


class AuditCache(TypedDict):
    """Snapshot of a full-library audit, keyed for fast re-filtering."""

    profile: BuildProfile
    issues: list[QualityIssue]
    coverage_counts: dict[str, list[int]]
    confirmed_candidates: list[str]
    entity_count: int
    confirmed_count: int
    draft_count: int
    deprecated_count: int
    snapshot_id: str
    source_snapshot_id: str


class EntitySnapshotRecord(TypedDict):
    """One readable formal entity or Symptom Profile in the build snapshot."""

    concept: str
    class_name: str
    object_name: str
    uri: str
    kind: str
    path: str
    parent_uri: str
    publication_state: str
    validation_state: str
    review_state: str
    content_hash: str
    content: str


class AuditMixin(WikiBuildDeps):
    """Full-library audit and paginated issue reporting."""

    def _formal_entity_snapshot_records(
        self, concept_filter: str = ""
    ) -> list[EntitySnapshotRecord]:
        """Enumerate the same formal pages used by audit and finalize.

        Symptom Profiles are physical child pages, not top-level entities.
        They must nevertheless be part of the publication snapshot with their
        real path; otherwise audit and finalize count different populations.
        """
        records: list[EntitySnapshotRecord] = []
        for concept, class_name, object_name, uri in self.store.list_entities():
            if concept_filter and concept != concept_filter:
                continue
            content = self.store.read_entity_by_uri(uri)
            if content is None:
                continue
            path = self.store.entity_path(
                concept,
                class_name or None,
                object_name,
            ).relative_to(self.store.root)
            records.append(
                {
                    "concept": concept,
                    "class_name": class_name or "",
                    "object_name": object_name,
                    "uri": uri,
                    "kind": "entity",
                    "path": str(path),
                    "parent_uri": "",
                    "publication_state": str(
                        parse_frontmatter(content).get("publication_state", "")
                    ),
                    "validation_state": str(parse_frontmatter(content).get("validation_state", "")),
                    "review_state": str(parse_frontmatter(content).get("review_state", "")),
                    "content_hash": sha256(content.encode("utf-8")).hexdigest(),
                    "content": content,
                },
            )
            if concept != "Symptom":
                continue
            for profile_id, profile_uri in self.store.list_symptom_profiles(uri):
                profile_path = self.store.symptom_profile_path(uri, profile_id)
                if not profile_path.is_file():
                    continue
                profile_content = profile_path.read_text(encoding="utf-8")
                records.append(
                    {
                        "concept": "Symptom",
                        "class_name": class_name or "",
                        "object_name": profile_id,
                        "uri": profile_uri,
                        "kind": "profile",
                        "path": str(profile_path.relative_to(self.store.root)),
                        "parent_uri": uri,
                        "publication_state": str(
                            parse_frontmatter(profile_content).get("publication_state", "")
                        ),
                        "validation_state": str(
                            parse_frontmatter(profile_content).get("validation_state", "")
                        ),
                        "review_state": str(
                            parse_frontmatter(profile_content).get("review_state", "")
                        ),
                        "content_hash": sha256(profile_content.encode("utf-8")).hexdigest(),
                        "content": profile_content,
                    },
                )
        return sorted(records, key=lambda record: record["uri"])

    def _tracked_gap_covers_issue(
        self,
        uri: str,
        message: str,
        pending_records: list[dict] | None = None,
        tracked_cache: dict[str, bool] | None = None,
    ) -> bool:
        """Downgrade only an explicitly marked, OPA-backed source gap."""
        records = pending_records
        if records is None:
            records = [
                *self.get_opas(target_uri=uri, category="gap", status="pending"),
                *self.get_opas(target_uri=uri, category="conflict", status="pending"),
            ]
        records = [record for record in records if str(record.get("target_uri", "")) == uri]
        if not records:
            return False
        lowered_message = message.lower()
        matching = [
            record
            for record in records
            if (
                tracked_cache.setdefault(
                    str(record.get("opa_id", record.get("uri", ""))),
                    self._is_explicit_tracked_record(record),
                )
                if tracked_cache is not None
                else self._is_explicit_tracked_record(record)
            )
            and any(
                name.lower() in lowered_message
                for name in opa_section_names(
                    str(record.get("target_section", "")),
                    " ".join(
                        str(record.get(field, ""))
                        for field in ("description", "finding", "missing", "recommendation")
                    ),
                )
            )
        ]
        return bool(matching)

    def audit_wiki(
        self,
        *,
        concept: str = "",
        code: str = "",
        offset: int = 0,
        limit: int = 100,
        profile: BuildProfile = "manual",
        entity_uris: list[str] | None = None,
        force_refresh: bool = False,
    ) -> WikiAuditReport:
        """Audit formal entities with a paginated actionable issue list.

        When ``entity_uris`` is provided, only those entities are audited
        (incremental mode).  Otherwise all entities (optionally filtered by
        ``concept``) are scanned.

        ``force_refresh`` bypasses the in-memory audit cache.  Use it when
        workers may have modified entities since the last full audit on this
        instance (e.g. before finalize).
        """
        if offset < 0:
            raise ValueError("audit_wiki offset must be non-negative")
        if limit < 1 or limit > 500:
            raise ValueError("audit_wiki limit must be between 1 and 500")
        audit_profile = profile
        if audit_profile not in BUILD_PROFILES:
            raise ValueError(
                f"audit_wiki profile must be one of: {', '.join(sorted(BUILD_PROFILES))}"
            )
        concept_filter = concept
        code_filter = code

        # ponytail: stale-cache guard — the cache is per-instance and only
        # invalidated by writes on THIS instance.  A conductor that never
        # writes will serve a frozen snapshot forever.  Cheap entity-count
        # probe (directory listing with its own TTL cache) catches adds/deletes;
        # force_refresh covers content-only patches.
        if force_refresh and self._audit_cache is not None:
            self._audit_cache = None
        if (
            entity_uris is None
            and self._audit_cache is not None
            and self._audit_cache["profile"] == audit_profile
        ):
            live_count = len(self.store.list_entities())
            if live_count != self._audit_cache["entity_count"]:
                self._audit_cache = None
            return self._finalize_audit_report(
                issues=self._audit_cache["issues"],
                coverage_counts=self._audit_cache["coverage_counts"],
                confirmed_candidates=self._audit_cache["confirmed_candidates"],
                entity_count=self._audit_cache["entity_count"],
                confirmed_count=self._audit_cache["confirmed_count"],
                draft_count=self._audit_cache["draft_count"],
                deprecated_count=self._audit_cache["deprecated_count"],
                snapshot_id=self._audit_cache["snapshot_id"],
                source_snapshot_id=self._audit_cache["source_snapshot_id"],
                concept_filter=concept_filter,
                code_filter=code_filter,
                offset=offset,
                limit=limit,
                audit_profile=audit_profile,
            )
        issues: list[QualityIssue] = []
        coverage_counts: dict[str, list[int]] = {}
        confirmed_candidates: list[str] = []
        entity_count = 0
        confirmed_count = 0
        draft_count = 0
        deprecated_count = 0

        snapshot_records = self._formal_entity_snapshot_records(concept_filter)
        snapshot_by_uri = {record["uri"]: record for record in snapshot_records}
        # A full audit can derive both directions from the in-memory snapshot.
        # A concept shard must not read every unrelated page: relation_worker
        # rebuilds the persisted backlink index before audit, so filtered
        # audits combine the selected pages' forward links with that index.
        graph_records = snapshot_records
        graph_uris = {
            uri for _concept, _class_name, _object_name, uri in self.store.list_entities()
        }
        graph_uris.update(record["uri"] for record in snapshot_records)
        # Concepts already materialized in this build. A relationship check
        # whose target concept exists must be repaired (build the link), not
        # recorded as a permanent content gap (open_gap / OPA).
        root_prefix = self.store.root_uri.rstrip("/") + "/"
        materialized_concepts = {
            uri.removeprefix(root_prefix).split("/", 1)[0]
            for uri in graph_uris
            if uri.startswith(root_prefix)
        }
        incoming_links: Counter[str] = Counter()
        outgoing_links: dict[str, set[str]] = {}
        for graph_record in graph_records:
            source_uri = graph_record["uri"]
            targets = {
                target.partition("#")[0]
                for target in extract_wiki_uris(graph_record["content"])
                if target.partition("#")[0] in graph_uris and target.partition("#")[0] != source_uri
            }
            outgoing_links[source_uri] = targets
            incoming_links.update(targets)
        if concept_filter:
            for record in snapshot_records:
                target_uri = record["uri"]
                incoming_links[target_uri] = sum(
                    1
                    for source_uri in self.store.get_backlinks(target_uri)
                    if source_uri != target_uri and source_uri in graph_uris
                )
        pending_opa_records = [
            *self.get_opas(category="gap", status="pending", limit=10000),
            *self.get_opas(category="conflict", status="pending", limit=10000),
        ]
        pending_opa_by_target: dict[str, list[dict]] = {}
        for record in pending_opa_records:
            pending_opa_by_target.setdefault(str(record.get("target_uri", "")), []).append(record)
        tracked_opa_cache: dict[str, bool] = {}
        resource_cache: dict[str, str | None] = {}
        raw_source_cache: dict[str, SourceReadResult] = {}
        source_hashes: dict[str, str] = {}

        def read_resource_cached(resource_uri: str) -> str | None:
            """Avoid rereading the same raw/wiki resource during one audit."""
            if resource_uri not in resource_cache:
                resource_cache[resource_uri] = self.read_resource(resource_uri)
            return resource_cache[resource_uri]

        def read_raw_source_cached(resource_uri: str) -> SourceReadResult:
            """Resolve provenance once and retain hashes for the audit snapshot."""
            if resource_uri not in raw_source_cache:
                raw_source_cache[resource_uri] = self.read_raw_source(resource_uri)
            result = raw_source_cache[resource_uri]
            if result.status is SourceReadStatus.OK and result.content_hash is not None:
                source_hashes[resource_uri] = result.content_hash
            return result

        entities: list[tuple[str, str, str, str, str]] = []
        if entity_uris is not None and entity_uris:
            for target_uri in entity_uris:
                record = snapshot_by_uri.get(target_uri)
                if record is None:
                    issues.append(
                        {
                            "uri": target_uri,
                            "concept": "",
                            "code": "entity_target_missing",
                            "severity": "error",
                            "message": "请求的实体 URI 不在存储快照中。",
                        },
                    )
                    continue
                entities.append(
                    (
                        record["concept"],
                        record["class_name"],
                        record["object_name"],
                        record["uri"],
                        record["content"],
                    ),
                )
        else:
            for record in snapshot_records:
                if concept_filter and record["concept"] != concept_filter:
                    continue
                entities.append(
                    (
                        record["concept"],
                        record["class_name"],
                        record["object_name"],
                        record["uri"],
                        record["content"],
                    ),
                )

        audit_hooks = tuple(
            hook
            for hook in ENTITY_VALIDATION_HOOKS
            if hook.name not in {"diagnostic_closure", "relationship_completeness", "body_sections"}
        )
        snapshot_id = sha256(
            "\n".join(
                f"{uri}\x1f{sha256(content.encode()).hexdigest()}"
                for _concept, _class_name, _object_name, uri, content in sorted(
                    entities, key=lambda item: item[3]
                )
            ).encode(),
        ).hexdigest()
        for concept, class_name, object_name, uri, content in entities:
            entity_count += 1
            _issue_start = len(issues)
            status = entity_status(content)
            if status == "confirmed":
                confirmed_count += 1
            elif status == "deprecated":
                deprecated_count += 1
                continue
            else:
                draft_count += 1
                issues.append(
                    {
                        "uri": uri,
                        "concept": concept,
                        "code": "unconfirmed_entity",
                        "severity": "warning",
                        "message": ("实体在所有确定性确认检查通过前仍为草稿状态。"),
                    },
                )

            entity_has_error = False
            frontmatter = parse_frontmatter(content)
            relation_concept = "SymptomProfile" if "profile_id" in frontmatter else concept
            for field, expected_concept in _TYPED_RELATION_FIELDS.get(relation_concept, {}).items():
                for target_uri in _resource_uris(frontmatter.get(field)):
                    target_base = target_uri.partition("#")[0]
                    target_identity = self.store.lookup_by_uri(target_base)
                    if target_identity is None or read_resource_cached(target_uri) is None:
                        entity_has_error = True
                        issues.append(
                            {
                                "uri": uri,
                                "concept": concept,
                                "code": "dangling_relation_target",
                                "severity": "error",
                                "message": f"关系字段 {field} 指向无法解析的 URI：{target_uri}",
                            },
                        )
                        continue
                    if target_identity[0] == expected_concept:
                        continue
                    entity_has_error = True
                    issues.append(
                        {
                            "uri": uri,
                            "concept": concept,
                            "code": "typed_relation_wrong_concept",
                            "severity": "error",
                            "message": (
                                f"关系字段 {field} 应指向 {expected_concept}，但实际指向 {target_identity[0]}：{target_uri}"
                            ),
                        },
                    )

            # Build-time writes may contain forward references, but a full
            # audit must not promote a page whose body still points at a
            # missing entity.  Validate profile links and heading fragments
            # through read_resource so Symptom Profile children are handled
            # as well as ordinary entities.
            for target_uri in extract_wiki_uris(content):
                target_base = target_uri.partition("#")[0]
                if target_base == uri:
                    continue
                if read_resource_cached(target_uri) is not None:
                    continue
                entity_has_error = True
                issues.append(
                    {
                        "uri": uri,
                        "concept": concept,
                        "code": "dangling_wiki_reference",
                        "severity": "error",
                        "message": f"正文引用无法解析：{target_uri}",
                    },
                )

            if concept == "Procedure":
                for specification_uri in _resource_uris(frontmatter.get("specification_refs")):
                    component_uri, separator, fragment = specification_uri.partition("#")
                    target_identity = self.store.lookup_by_uri(component_uri)
                    specification_issue = ""
                    if target_identity is None or target_identity[0] != "Component":
                        specification_issue = "流程 specification_ref 必须指向已存在的部件。"
                    elif not separator or not fragment.strip():
                        specification_issue = (
                            "流程 specification_ref 必须包含 Component#spec 片段。"
                        )
                    elif read_resource_cached(specification_uri) is None:
                        specification_issue = (
                            f"流程 specification 片段无法解析：{specification_uri}"
                        )
                    if specification_issue:
                        entity_has_error = True
                        issues.append(
                            {
                                "uri": uri,
                                "concept": concept,
                                "code": "Procedure.specification_ref_unresolvable",
                                "severity": "error",
                                "message": specification_issue,
                            },
                        )

            if relation_concept == "SymptomProfile":
                parent_symptom = frontmatter.get("parent_symptom")
                direct_component = _resource_uris(frontmatter.get("direct_component_uri"))
                for device_uri in _resource_uris(frontmatter.get("device_refs")):
                    device_content = read_resource_cached(device_uri)
                    if device_content is None:
                        continue
                    device_frontmatter = parse_frontmatter(device_content)
                    critical_components = _resource_uris(
                        device_frontmatter.get("critical_components")
                    )
                    if (
                        critical_components
                        and direct_component
                        and direct_component[0] not in critical_components
                    ):
                        entity_has_error = True
                        issues.append(
                            {
                                "uri": uri,
                                "concept": concept,
                                "code": "Profile.direct_component_not_in_device_bom",
                                "severity": "error",
                                "message": (
                                    f"画像直接部件 {direct_component[0]} 未列入设备 {device_uri} 的关重件清单。"
                                ),
                            },
                        )
                    if isinstance(parent_symptom, str):
                        symptom_refs = _resource_uris(device_frontmatter.get("symptom_refs"))
                        if symptom_refs and parent_symptom not in symptom_refs:
                            entity_has_error = True
                            issues.append(
                                {
                                    "uri": uri,
                                    "concept": concept,
                                    "code": "Profile.parent_symptom_not_indexed_by_device",
                                    "severity": "error",
                                    "message": f"设备 {device_uri} 未索引父级故障现象 {parent_symptom}。",
                                },
                            )
            declared_sources = list(
                dict.fromkeys(
                    _raw_source_uris(
                        frontmatter.get("sources"),
                        raw_root_uri=self._raw_fs.root_uri,
                    ),
                ),
            )
            body_sources = [
                source_uri
                for source_uri in extract_source_uris(content)
                if classify_raw_source_uri(
                    source_uri,
                    raw_root_uri=self._raw_fs.root_uri,
                )
                is not None
            ]
            source_uris = list(dict.fromkeys([*declared_sources, *body_sources]))

            for source_uri in source_uris:
                source_result = read_raw_source_cached(source_uri)
                if source_result.status is SourceReadStatus.OK:
                    continue
                entity_has_error = True
                issues.append(
                    {
                        "uri": uri,
                        "concept": concept,
                        "code": "source_unresolvable",
                        "severity": "error",
                        "message": f"原始引用无法解析：{source_uri}",
                    },
                )

            for requirement in confirmation_requirements(
                content,
                concept,
                class_name,
            ):
                counts = coverage_counts.setdefault(requirement.code, [0, 0])
                counts[0] += 1
                if requirement.complete:
                    counts[1] += 1
                    continue
                disposition = requirement.disposition.value
                if requirement.target_concepts and materialized_concepts.intersection(
                    requirement.target_concepts
                ):
                    disposition = IssueDisposition.REPAIR_ONLY.value
                severity = (
                    "warning"
                    if audit_profile == "case" and requirement.code in _CASE_SOFT_REQUIREMENT_CODES
                    else "error"
                )
                # REPAIR_ONLY means the target concept is already materialized:
                # the missing link/entity can and must be built. A pending OPA
                # must never downgrade that into a warning, or finalize would
                # publish the page with an unfixed relationship.
                if (
                    severity == "error"
                    and disposition != IssueDisposition.REPAIR_ONLY.value
                    and self._tracked_gap_covers_issue(
                        uri,
                        f"{requirement.code}: {requirement.message}",
                        pending_opa_by_target.get(uri, []),
                        tracked_opa_cache,
                    )
                ):
                    severity = "warning"
                if severity == "error":
                    entity_has_error = True
                issue: QualityIssue = {
                    "uri": uri,
                    "concept": concept,
                    "code": requirement.code,
                    "severity": severity,
                    "message": requirement.message,
                    "disposition": disposition,
                    "opa_reason_code": requirement.opa_reason_code,
                }
                if disposition == IssueDisposition.REPAIR_ONLY.value:
                    issue["repair_action"] = _repair_action_for(
                        requirement.code, requirement.target_concepts
                    )
                issues.append(issue)

            if has_unresolved_placeholder(content):
                entity_has_error = True
                issues.append(
                    {
                        "uri": uri,
                        "concept": concept,
                        "code": "unresolved_placeholder",
                        "severity": "error",
                        "message": ("正式内容仍包含未解决的 agent TODO 或关系占位符。"),
                    },
                )

            validation_results = run_entity_validation(
                content=force_confirmed_status(content),
                concept=concept,
                class_name=class_name,
                object_name=object_name,
                hooks=audit_hooks,
            )
            for result in validation_results:
                if result.passed:
                    continue
                # ponytail: content-grade hooks (body evidence, source refs,
                # lightweight materialization) are structural gates only at
                # finalize; on draft entities demote to warning so mid-build
                # audits surface them as backlog instead of blocking the loop.
                # Structural hooks (reference integrity, directory structure,
                # taxonomy, schema, URI) stay errors at every stage.
                severity = result.severity
                message = result.message
                if (
                    status != "confirmed"
                    and result.severity == "error"
                    and result.hook_name
                    in (
                        "body_evidence",
                        "body_sections",
                        "source_ref",
                        "lightweight_materialization",
                    )
                ):
                    severity = "warning"
                    message = f"[draft] {message}"
                if severity == "error":
                    entity_has_error = True
                issues.append(
                    {
                        "uri": uri,
                        "concept": concept,
                        "code": f"hook.{result.hook_name}",
                        "severity": severity,
                        "message": message,
                    },
                )

            for target_uri in extract_wiki_uris(content):
                if read_resource_cached(target_uri) is not None:
                    continue
                entity_has_error = True
                issues.append(
                    {
                        "uri": uri,
                        "concept": concept,
                        "code": "dangling_reference",
                        "severity": "error",
                        "message": f"引用资源无法解析：{target_uri}",
                    },
                )

            for bad_uri in extract_malformed_wiki_uris(content):
                entity_has_error = True
                issues.append(
                    {
                        "uri": uri,
                        "concept": concept,
                        "code": "malformed_uri",
                        "severity": "error",
                        "message": f"wiki URI 格式错误（空/占位符/无对象尾段）：{self.store.root_uri}/.../{bad_uri or '(空)'}",
                    },
                )

            # ponytail: draft entities with structural errors (hook.*,
            # dangling refs, malformed URIs, etc.) classify as
            # data_limitation so finalize proceeds — the entity stays
            # draft (entity_has_error=True) and issues remain visible.
            # Confirmed entities keep dependency_gap → finalize blocks.
            if status != "confirmed":
                for _issue in issues[_issue_start:]:
                    if _issue.get("severity") == "error" and "gap_category" not in _issue:
                        _issue["gap_category"] = "data_limitation"

            if not entity_has_error and status != "confirmed":
                confirmed_candidates.append(uri)

        if entity_count == 0:
            issues.append(
                {
                    "uri": self.store.root_uri + "/",
                    "concept": "",
                    "code": "empty_wiki",
                    "severity": "error",
                    "message": "空 wiki 无法完成定稿。",
                },
            )

        for issue in issues:
            if "disposition" in issue:
                continue
            policy = audit_issue_policy(issue["code"])
            if policy is None:
                continue
            issue["disposition"] = policy.disposition.value
            if policy.opa_reason_code:
                issue["opa_reason_code"] = policy.opa_reason_code

        source_snapshot_id = sha256(
            "\n".join(
                f"{source_uri}\x1f{content_hash}"
                for source_uri, content_hash in sorted(source_hashes.items())
            ).encode(),
        ).hexdigest()

        relation_coverage: dict[str, CoverageSummary] = {}
        for relation_code, counts in sorted(coverage_counts.items()):
            eligible, complete = counts
            relation_coverage[relation_code] = {
                "eligible": eligible,
                "complete": complete,
                "percent": round((complete / eligible) * 100, 2),
            }

        # Only an unfiltered scan is a valid reusable snapshot.  A
        # concept-filtered audit is intentionally a shard and must never
        # replace the cache: doing so makes the next concept appear to have
        # zero issues and can create a false finalize decision.
        if entity_uris is None and not concept_filter and self._audit_cache is None:
            snapshot: AuditCache = {
                "profile": audit_profile,
                "issues": issues,
                "coverage_counts": coverage_counts,
                "confirmed_candidates": confirmed_candidates,
                "entity_count": entity_count,
                "confirmed_count": confirmed_count,
                "draft_count": draft_count,
                "deprecated_count": deprecated_count,
                "snapshot_id": snapshot_id,
                "source_snapshot_id": source_snapshot_id,
            }
            self._audit_cache = snapshot
        return self._finalize_audit_report(
            issues=issues,
            coverage_counts=coverage_counts,
            confirmed_candidates=confirmed_candidates,
            entity_count=entity_count,
            confirmed_count=confirmed_count,
            draft_count=draft_count,
            deprecated_count=deprecated_count,
            snapshot_id=snapshot_id,
            source_snapshot_id=source_snapshot_id,
            concept_filter=concept_filter,
            code_filter=code_filter,
            offset=offset,
            limit=limit,
            audit_profile=audit_profile,
        )

    def _finalize_audit_report(
        self,
        *,
        issues: list[QualityIssue],
        coverage_counts: dict[str, list[int]],
        confirmed_candidates: list[str],
        entity_count: int,
        confirmed_count: int,
        draft_count: int,
        deprecated_count: int,
        snapshot_id: str = "",
        source_snapshot_id: str = "",
        concept_filter: str,
        code_filter: str,
        offset: int,
        limit: int,
        audit_profile: str,
    ) -> WikiAuditReport:
        """Build the paginated audit report from full issue/coverage data."""
        relation_coverage: dict[str, CoverageSummary] = {}
        for relation_code, counts in sorted(coverage_counts.items()):
            eligible, complete = counts
            relation_coverage[relation_code] = {
                "eligible": eligible,
                "complete": complete,
                "percent": round((complete / eligible) * 100, 2),
            }
        error_count = sum(issue["severity"] == "error" for issue in issues)
        warning_count = len(issues) - error_count
        issue_counts = dict(Counter(issue["code"] for issue in issues).most_common())
        filtered_issues = [
            issue
            for issue in issues
            if (not concept_filter or issue["concept"] == concept_filter)
            and (not code_filter or issue["code"] == code_filter)
        ]
        returned_issues = filtered_issues[offset : offset + limit]
        for issue in returned_issues:
            if "target_uri" not in issue:
                issue["target_uri"] = issue.get("uri", "")
            if "gap_category" not in issue:
                issue["gap_category"] = _classify_gap_category(
                    issue.get("code", ""),
                    issue.get("opa_reason_code", ""),
                    issue.get("disposition", ""),
                )
        next_offset = offset + limit if offset + limit < len(filtered_issues) else -1
        return {
            # ``status=draft`` is a legacy build marker, not a human-review
            # gate.  Finalize admits only citation/structure-clean pages (or
            # explicitly OPA-backed open gaps) and stamps the independent
            # publication/review fields after that gate.
            "passed": error_count == 0 and entity_count > 0,
            "profile": audit_profile,
            "entity_count": entity_count,
            "confirmed_count": confirmed_count,
            "draft_count": draft_count,
            "deprecated_count": deprecated_count,
            "snapshot_id": snapshot_id,
            "source_snapshot_id": source_snapshot_id,
            "error_count": error_count,
            "warning_count": warning_count,
            "issue_counts": issue_counts,
            "filtered_issue_count": len(filtered_issues),
            "returned_issue_count": len(returned_issues),
            "next_offset": next_offset,
            "relation_coverage": relation_coverage,
            "confirmed_candidates": confirmed_candidates,
            "issues": returned_issues,
        }
