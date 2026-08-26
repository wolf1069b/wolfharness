"""Children/parent navigation mixin for WikiBuildTools — hierarchical URI drill-down."""

from __future__ import annotations

import re

from wolfharness.capabilities.wiki.quality import (
    extract_sections,
    extract_source_uris,
    extract_wiki_uris,
    has_usable_procedure_criteria,
    is_external_source_uri,
    is_raw_chapter_uri as is_registered_raw_chapter_uri,
    is_source_uri_scheme,
    parse_frontmatter,
)
from wolfharness.capabilities.wiki.storage import viking_list_children
from wolfharness.capabilities.wiki.wiki_build_deps import WikiBuildDeps


_RELATION_FIELDS: dict[str, tuple[str, ...]] = {
    "Device": ("critical_components", "symptom_refs"),
    "Component": ("assembly_parts",),
    "SymptomProfile": ("device_refs", "direct_component_uri", "possible_faults"),
    "Fault": (
        "affected_components",
        "verification_procedures",
        "repair_procedures",
    ),
    "DTC": ("controller_component", "related_faults"),
    "Procedure": ("target_components", "specification_refs"),
}

_TOP_LEVEL_STEP_RE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.+?)\s*$")
_TRACE_GAP_RE = re.compile(r"(?:open_gap|待补充|来源未说明|未物化|来源缺失)", re.IGNORECASE)


def _uri_values(root_uri: str, value: object) -> list[str]:
    """Collect wiki URIs from a YAML scalar/list/map without trusting names."""
    prefix = root_uri + "/"
    if isinstance(value, str):
        return [value.strip()] if value.strip().startswith(prefix) else []
    if isinstance(value, dict):
        result: list[str] = []
        for nested in value.values():
            result.extend(_uri_values(root_uri, nested))
        return result
    if isinstance(value, (list, tuple, set)):
        result = []
        for nested in value:
            result.extend(_uri_values(root_uri, nested))
        return result
    return []


def _source_values(value: object) -> list[str]:
    """Collect source/resource URIs from frontmatter values."""
    if isinstance(value, str):
        return [value.strip()] if is_source_uri_scheme(value.strip()) else []
    if isinstance(value, dict):
        result: list[str] = []
        for nested in value.values():
            result.extend(_source_values(nested))
        return result
    if isinstance(value, (list, tuple, set)):
        result = []
        for nested in value:
            result.extend(_source_values(nested))
        return result
    return []


def _section_at(content: str, position: int) -> tuple[str, str]:
    """Return the nearest ``##`` heading and its state for a body position."""
    body_start = content.find("\n---\n")
    body = content[body_start + 5 :] if body_start >= 0 else content
    body_offset = body_start + 5 if body_start >= 0 else 0
    local_position = max(0, position - body_offset)
    heading = ""
    heading_position = -1
    for match in re.finditer(r"^##\s+(.+?)\s*$", body, re.MULTILINE):
        if match.start() > local_position:
            break
        heading = match.group(1).strip()
        heading_position = match.end()
    next_heading = re.search(r"^##\s+.+?$", body[local_position:], re.MULTILINE)
    section_text = body[
        heading_position : local_position + (next_heading.start() if next_heading else len(body))
    ]
    lowered = section_text.lower()
    if any(marker in lowered for marker in ("open_gap", "待补充", "缺失", "未提供")):
        state = "open_gap"
    elif "冲突" in section_text:
        state = "conflict_pending"
    else:
        state = "complete"
    return heading, state


def _is_raw_chapter_uri(uri: str) -> bool:
    """Return whether ``uri`` is a resolvable source citation.

    Recognizes real chapter paths (``{root}/<doc>/chapters/<subdir>/chapter.md``)
    and external MCP source URIs (any non-viking/non-file scheme).
    """
    return is_registered_raw_chapter_uri(uri) or is_external_source_uri(uri)


def _raw_source_values(values: list[str]) -> list[str]:
    """Keep chapter citations that this trace can resolve through ``read_resource``."""
    return [uri for uri in dict.fromkeys(values) if _is_raw_chapter_uri(uri)]


def _content_source_values(content: str, frontmatter: dict[str, object]) -> list[str]:
    """Collect page and body raw citations without inventing provenance."""
    return list(
        dict.fromkeys(
            [
                *_source_values(frontmatter.get("sources")),
                *(uri for uri in extract_source_uris(content) if _is_raw_chapter_uri(uri)),
            ],
        ),
    )


def _diagnostic_step_items(section: str) -> list[str]:
    """Return top-level diagnostic steps without interpreting their prose."""
    return [
        match.group(1).strip()
        for line in section.splitlines()
        if (match := _TOP_LEVEL_STEP_RE.match(line)) is not None
    ]


def _step_evidence(section: str) -> list[dict[str, object]]:
    """Return raw evidence attached to each numbered/list procedure step."""
    evidence: list[dict[str, object]] = []
    for step in _diagnostic_step_items(section):
        direct_sources = [uri for uri in extract_source_uris(step) if _is_raw_chapter_uri(uri)]
        evidence.append(
            {
                "step": step,
                "raw_source_uris": direct_sources,
                "effective_source_uris": direct_sources,
                "has_local_evidence": bool(direct_sources),
            },
        )
    return evidence


class ChildrenMixin(WikiBuildDeps):
    """Unified hierarchical browsing by URI (wiki entities, case files, fault-annotated docs)."""

    def _discover_devices_for_symptom(self, symptom_uri: str) -> list[dict[str, object]]:
        """Find model pages that explicitly index a Symptom or its Profiles."""
        discovered: dict[str, dict[str, object]] = {}
        for concept, class_name, object_name, device_uri in self.store.list_entities("Device"):
            if concept != "Device":
                continue
            content = self.read_resource(device_uri)
            if content is None:
                continue
            frontmatter = parse_frontmatter(content)
            if symptom_uri not in _uri_values(self.store.root_uri, frontmatter.get("symptom_refs")):
                continue
            discovered[device_uri] = {
                "device_uri": device_uri,
                "indexed_by_device": True,
                "indexed_by_profile": False,
                "class_name": class_name,
                "object_name": object_name,
            }

        for profile in self.list_symptom_profiles(symptom_uri):
            profile_content = self.read_resource(profile["uri"])
            if profile_content is None:
                continue
            for device_uri in _uri_values(
                self.store.root_uri, parse_frontmatter(profile_content).get("device_refs")
            ):
                record = discovered.setdefault(
                    device_uri,
                    {
                        "device_uri": device_uri,
                        "indexed_by_device": False,
                        "indexed_by_profile": False,
                    },
                )
                record["indexed_by_profile"] = True

        return [discovered[uri] for uri in sorted(discovered)]

    def trace_diagnostic_path(
        self,
        device_uri: str = "",
        symptom_uri: str = "",
        limit: int = 100,
    ) -> dict[str, object]:
        """Trace Device → Symptom/Profile → Fault → Component → Procedure.

        The trace only follows typed frontmatter relations.  Narrative names
        and ``applicable_models`` strings are deliberately not used as
        substitutes, so missing bindings remain visible to the caller.
        """
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        if not device_uri:
            if not symptom_uri:
                raise ValueError("device_uri or symptom_uri must be provided")
            discovery = self._discover_devices_for_symptom(symptom_uri)
            if not discovery:
                return {
                    "symptom_uri": symptom_uri,
                    "device_uris": [],
                    "device_discovery": [],
                    "paths": [],
                    "complete": False,
                    "error": "device_not_discovered_for_symptom",
                }
            truncated = {"devices": len(discovery)} if len(discovery) > limit else {}
            paths = [
                self.trace_diagnostic_path(
                    str(record["device_uri"]),
                    symptom_uri=symptom_uri,
                    limit=limit,
                )
                for record in discovery[:limit]
            ]
            return {
                "symptom_uri": symptom_uri,
                "device_uris": [str(record["device_uri"]) for record in discovery[:limit]],
                "device_discovery": discovery[:limit],
                "paths": paths,
                "truncated": truncated,
                "complete": (
                    bool(paths)
                    and not truncated
                    and all(bool(path.get("complete")) for path in paths)
                ),
            }
        device_content = self.read_resource(device_uri)
        if device_content is None:
            return {"device_uri": device_uri, "complete": False, "error": "device_not_found"}
        device_frontmatter = parse_frontmatter(device_content)
        symptom_uris = _uri_values(self.store.root_uri, device_frontmatter.get("symptom_refs"))
        device_components = set(
            _uri_values(self.store.root_uri, device_frontmatter.get("critical_components"))
        )
        device_sections = extract_sections(device_content)
        device_body_link_uris = sorted(extract_wiki_uris(device_content))
        device_typed_relations = {
            field: _uri_values(self.store.root_uri, device_frontmatter.get(field))
            for field in _RELATION_FIELDS["Device"]
            if _uri_values(self.store.root_uri, device_frontmatter.get(field))
        }
        device_related_uris = sorted(
            set(device_body_link_uris).union(
                uri for values in device_typed_relations.values() for uri in values
            ),
        )
        truncated: dict[str, int] = {}
        if len(device_components) > limit:
            truncated["critical_components"] = len(device_components)
        if len(device_body_link_uris) > limit:
            truncated["device_body_links"] = len(device_body_link_uris)
        if len(symptom_uris) > limit:
            truncated["symptoms"] = len(symptom_uris)
        device_typed_uri_set = {uri for values in device_typed_relations.values() for uri in values}
        device_extra_resources: list[dict[str, object]] = []
        device_extra_unresolved: list[str] = []
        device_extra_source_uris: list[str] = []
        for linked_uri in device_body_link_uris[:limit]:
            if linked_uri == device_uri or linked_uri in device_typed_uri_set:
                continue
            linked_content = self.read_resource(linked_uri)
            if linked_content is None:
                device_extra_unresolved.append(linked_uri)
                device_extra_resources.append(
                    {"uri": linked_uri, "complete": False, "error": "resource_not_found"},
                )
                continue
            linked_frontmatter = parse_frontmatter(linked_content)
            linked_sources = _content_source_values(linked_content, linked_frontmatter)
            linked_unresolved_sources = [
                uri for uri in _raw_source_values(linked_sources) if self.read_resource(uri) is None
            ]
            device_extra_source_uris.extend(linked_unresolved_sources)
            device_extra_resources.append(
                {
                    "uri": linked_uri,
                    "content": linked_content,
                    "source_uris": linked_sources,
                    "unresolved_source_uris": linked_unresolved_sources,
                    "complete": not linked_unresolved_sources,
                },
            )
        device_source_uris = _content_source_values(device_content, device_frontmatter)
        unresolved_source_uris = [
            uri for uri in _raw_source_values(device_source_uris) if self.read_resource(uri) is None
        ]
        unresolved_source_uris.extend(device_extra_source_uris)
        complete = bool(
            symptom_uris
            and device_components
            and _raw_source_values(device_source_uris)
            and not unresolved_source_uris,
        )
        if device_extra_unresolved:
            complete = False
        component_resources: list[dict[str, object]] = []
        for component_uri in sorted(device_components)[:limit]:
            component_content = self.read_resource(component_uri)
            if component_content is None:
                complete = False
                component_resources.append(
                    {"uri": component_uri, "complete": False, "error": "component_not_found"},
                )
                continue
            component_frontmatter = parse_frontmatter(component_content)
            component_source_uris = _content_source_values(component_content, component_frontmatter)
            component_unresolved = [
                uri
                for uri in _raw_source_values(component_source_uris)
                if self.read_resource(uri) is None
            ]
            unresolved_source_uris.extend(component_unresolved)
            if not _raw_source_values(component_source_uris) or component_unresolved:
                complete = False
            component_resources.append(
                {
                    "uri": component_uri,
                    "title": component_frontmatter.get("title", ""),
                    "content": component_content,
                    "source_uris": component_source_uris,
                    "unresolved_source_uris": component_unresolved,
                    "complete": bool(component_source_uris and not component_unresolved),
                },
            )
        if symptom_uri:
            if symptom_uri not in symptom_uris:
                return {
                    "device_uri": device_uri,
                    "symptom_uri": symptom_uri,
                    "complete": False,
                    "error": "symptom_not_indexed_by_device",
                    "symptom_uris": symptom_uris,
                }
            symptom_uris = [symptom_uri]

        symptoms: list[dict[str, object]] = []
        procedure_uris: list[str] = []
        expected_procedure_components: dict[str, set[str]] = {}
        for current_symptom_uri in symptom_uris[:limit]:
            symptom_content = self.read_resource(current_symptom_uri)
            if symptom_content is None:
                complete = False
                symptoms.append({
                    "uri": current_symptom_uri,
                    "complete": False,
                    "error": "symptom_not_found",
                })
                continue
            symptom_frontmatter = parse_frontmatter(symptom_content)
            symptom_source_uris = _content_source_values(symptom_content, symptom_frontmatter)
            symptom_unresolved = [
                uri
                for uri in _raw_source_values(symptom_source_uris)
                if self.read_resource(uri) is None
            ]
            unresolved_source_uris.extend(symptom_unresolved)
            if not _raw_source_values(symptom_source_uris) or symptom_unresolved:
                complete = False
            profiles: list[dict[str, object]] = []
            for profile in self.list_symptom_profiles(current_symptom_uri):
                profile_uri = profile["uri"]
                profile_content = self.read_resource(profile_uri)
                if profile_content is None:
                    complete = False
                    profiles.append({
                        "uri": profile_uri,
                        "complete": False,
                        "error": "profile_not_found",
                    })
                    continue
                profile_frontmatter = parse_frontmatter(profile_content)
                profile_source_uris = _content_source_values(profile_content, profile_frontmatter)
                profile_unresolved = [
                    uri
                    for uri in _raw_source_values(profile_source_uris)
                    if self.read_resource(uri) is None
                ]
                unresolved_source_uris.extend(profile_unresolved)
                if not _raw_source_values(profile_source_uris) or profile_unresolved:
                    complete = False
                profile_devices = _uri_values(
                    self.store.root_uri, profile_frontmatter.get("device_refs")
                )
                if device_uri not in profile_devices:
                    continue
                component_uri = profile_frontmatter.get("direct_component_uri")
                component_uris = _uri_values(self.store.root_uri, component_uri)
                fault_uris = _uri_values(
                    self.store.root_uri, profile_frontmatter.get("possible_faults")
                )
                if len(fault_uris) > limit:
                    truncated[f"faults:{profile_uri}"] = len(fault_uris)
                if component_uris and not set(component_uris) <= device_components:
                    complete = False
                profile_sections = extract_sections(profile_content)
                diagnostic_section = profile_sections.get("推荐诊断流程", "")
                diagnostic_steps = _diagnostic_step_items(diagnostic_section)
                missing_procedure_steps = [
                    step
                    for step in diagnostic_steps
                    if not _TRACE_GAP_RE.search(step)
                    and not any(
                        uri.startswith(self.store.root_uri + "/Procedure/")
                        for uri in extract_wiki_uris(step)
                    )
                ]
                open_gap_steps = [step for step in diagnostic_steps if _TRACE_GAP_RE.search(step)]
                diagnostic_flow_checked = bool(diagnostic_section.strip())
                diagnostic_flow_complete = (
                    None
                    if not diagnostic_flow_checked
                    else bool(diagnostic_steps)
                    and not missing_procedure_steps
                    and not open_gap_steps
                )
                for procedure_uri in extract_wiki_uris(diagnostic_section):
                    if (
                        procedure_uri.startswith(self.store.root_uri + "/Procedure/")
                        and procedure_uri not in procedure_uris
                    ):
                        procedure_uris.append(procedure_uri)
                    if procedure_uri.startswith(self.store.root_uri + "/Procedure/"):
                        expected_procedure_components.setdefault(procedure_uri, set()).update(
                            component_uris
                        )
                faults: list[dict[str, object]] = []
                if not component_uris or not fault_uris:
                    complete = False
                for fault_uri in fault_uris[:limit]:
                    fault_content = self.read_resource(fault_uri)
                    if fault_content is None:
                        complete = False
                        faults.append({
                            "uri": fault_uri,
                            "complete": False,
                            "error": "fault_not_found",
                        })
                        continue
                    fault_frontmatter = parse_frontmatter(fault_content)
                    fault_source_uris = _content_source_values(fault_content, fault_frontmatter)
                    fault_unresolved = [
                        uri
                        for uri in _raw_source_values(fault_source_uris)
                        if self.read_resource(uri) is None
                    ]
                    unresolved_source_uris.extend(fault_unresolved)
                    if not _raw_source_values(fault_source_uris) or fault_unresolved:
                        complete = False
                    affected_components = _uri_values(
                        self.store.root_uri, fault_frontmatter.get("affected_components")
                    )
                    verification = _uri_values(
                        self.store.root_uri, fault_frontmatter.get("verification_procedures")
                    )
                    repair = _uri_values(
                        self.store.root_uri, fault_frontmatter.get("repair_procedures")
                    )
                    for procedure_uri in [*verification, *repair]:
                        if procedure_uri not in procedure_uris:
                            procedure_uris.append(procedure_uri)
                        expected_procedure_components.setdefault(procedure_uri, set()).update(
                            affected_components,
                        )
                    component_covered = bool(set(component_uris) & set(affected_components))
                    affected_component_in_bom = bool(set(affected_components) & device_components)
                    if (
                        not affected_components
                        or not component_covered
                        or not affected_component_in_bom
                        or (not verification and not repair)
                    ):
                        complete = False
                    faults.append(
                        {
                            "uri": fault_uri,
                            "component_uris": affected_components,
                            "verification_procedures": verification,
                            "repair_procedures": repair,
                            "direct_component_covered": component_covered,
                            "affected_component_in_bom": affected_component_in_bom,
                            "source_uris": fault_source_uris,
                            "unresolved_source_uris": fault_unresolved,
                            "content": fault_content,
                            "complete": bool(affected_components and (verification or repair)),
                        },
                    )
                profiles.append(
                    {
                        "uri": profile_uri,
                        "device_uris": profile_devices,
                        "direct_component_uri": component_uris,
                        "faults": faults,
                        "diagnostic_step_count": len(diagnostic_steps),
                        "linked_procedure_step_count": len(diagnostic_steps)
                        - len(missing_procedure_steps),
                        "missing_procedure_steps": missing_procedure_steps,
                        "open_gap_steps": open_gap_steps,
                        "diagnostic_flow_complete": diagnostic_flow_complete,
                        "source_uris": profile_source_uris,
                        "unresolved_source_uris": profile_unresolved,
                        "content": profile_content,
                        "complete": bool(
                            component_uris
                            and fault_uris
                            and faults
                            and all(bool(fault.get("complete")) for fault in faults)
                            and diagnostic_flow_complete is not False,
                        ),
                    },
                )
            if not profiles:
                complete = False
            symptoms.append({
                "uri": current_symptom_uri,
                "profiles": profiles,
                "complete": bool(profiles),
            })

        procedures: list[dict[str, object]] = []
        if len(procedure_uris) > limit:
            truncated["procedures"] = len(procedure_uris)
        for procedure_uri in procedure_uris[:limit]:
            procedure_content = self.read_resource(procedure_uri)
            if procedure_content is None:
                complete = False
                procedures.append({
                    "uri": procedure_uri,
                    "complete": False,
                    "error": "procedure_not_found",
                })
                continue
            procedure_frontmatter = parse_frontmatter(procedure_content)
            procedure_sections = extract_sections(procedure_content)
            operation_steps = procedure_sections.get("操作步骤", "")
            criteria = procedure_sections.get("判定标准", "")
            operation_sources = sorted(
                uri for uri in extract_source_uris(operation_steps) if _is_raw_chapter_uri(uri)
            )
            criteria_sources = sorted(
                uri for uri in extract_source_uris(criteria) if _is_raw_chapter_uri(uri)
            )
            procedure_source_uris = _content_source_values(procedure_content, procedure_frontmatter)
            procedure_unresolved = [
                uri
                for uri in _raw_source_values([
                    *procedure_source_uris,
                    *operation_sources,
                    *criteria_sources,
                ])
                if self.read_resource(uri) is None
            ]
            unresolved_source_uris.extend(procedure_unresolved)
            target_components = _uri_values(
                self.store.root_uri, procedure_frontmatter.get("target_components")
            )
            expected_components = expected_procedure_components.get(procedure_uri, set())
            target_is_in_device_bom = bool(set(target_components) & device_components)
            target_covers_chain = not expected_components or bool(
                set(target_components) & expected_components,
            )
            operation_evidence_sources = list(
                dict.fromkeys(operation_sources),
            )
            criteria_evidence_sources = list(
                dict.fromkeys(criteria_sources),
            )
            operation_step_evidence = _step_evidence(operation_steps)
            missing_operation_step_evidence = [
                str(item["step"])
                for item in operation_step_evidence
                if not bool(item["has_local_evidence"])
            ]
            procedure_complete = bool(
                target_components
                and target_is_in_device_bom
                and target_covers_chain
                and operation_steps.strip()
                and has_usable_procedure_criteria(criteria)
                and _raw_source_values(operation_evidence_sources)
                and _raw_source_values(criteria_evidence_sources)
                and not procedure_unresolved,
            )
            procedures.append(
                {
                    "uri": procedure_uri,
                    "target_components": target_components,
                    "target_is_in_device_bom": target_is_in_device_bom,
                    "target_covers_chain": target_covers_chain,
                    "source_uris": list(
                        dict.fromkeys(
                            [
                                *procedure_source_uris,
                                *operation_sources,
                                *criteria_sources,
                            ],
                        ),
                    ),
                    "operation_steps": operation_steps,
                    "operation_step_evidence": operation_step_evidence,
                    "missing_operation_step_evidence": missing_operation_step_evidence,
                    "operation_source_uris": operation_evidence_sources,
                    "criteria": criteria,
                    "criteria_usable": has_usable_procedure_criteria(criteria),
                    "criteria_source_uris": criteria_evidence_sources,
                    "unresolved_source_uris": procedure_unresolved,
                    "complete": procedure_complete,
                },
            )
            if not procedure_complete:
                complete = False

        if truncated:
            complete = False

        return {
            "device_uri": device_uri,
            "device": {
                "uri": device_uri,
                "content": device_content,
                "frontmatter": device_frontmatter,
                "sections": device_sections,
                "typed_relations": device_typed_relations,
                "body_link_uris": device_body_link_uris,
                "related_uris": device_related_uris,
                "extra_resources": device_extra_resources,
                "unresolved_related_uris": sorted(device_extra_unresolved),
                "source_uris": device_source_uris,
                "unresolved_source_uris": sorted(
                    set(_raw_source_values(device_source_uris)).union(device_extra_unresolved)
                    & set(unresolved_source_uris),
                ),
            },
            "symptom_uri": symptom_uri,
            "device_source_uris": device_source_uris,
            "critical_component_uris": sorted(device_components),
            "component_resources": component_resources,
            "symptoms": symptoms,
            "procedures": procedures,
            "truncated": truncated,
            "unresolved_source_uris": sorted(set(unresolved_source_uris)),
            "complete": complete,
            "indexed_symptom_count": len(symptom_uris),
        }

    def list_children(self, uri: str) -> dict:
        """Browse by URI — unified hierarchical drill-down.

        Supports three URI schemes:
        - ``{root_uri}/`` → wiki entities (concepts → classes → entities)
        - ``viking://resources/`` → arbitrary remote trees (raw manuals, cases)
        - ``{bom_root}/`` → the configured global BOM tree

        Wiki entities and classes use readable, self-describing paths.
        For BOM URIs, the configured global BOM tree is browsed directly.
        For viking URIs, the remote namespace is traversed via the SDK.
        """
        if self._bom_fs is not None:
            bom_root = self._bom_fs.root_uri.rstrip("/")
            if uri == bom_root or uri.startswith(bom_root + "/"):
                return self._list_bom_children(uri)
        if uri.startswith("viking://resources/") and not (
            uri == self.store.root_uri or uri.startswith(self.store.root_uri + "/")
        ):
            return viking_list_children(uri)

        path = uri.removeprefix(self.store.root_uri).strip("/")
        parts = [p for p in path.split("/") if p]

        if not parts:
            entities = self.store.list_entities()
            counts: dict[str, int] = {}
            for c, _cls, _obj, _u in entities:
                counts[c] = counts.get(c, 0) + 1
            return {
                "uri": uri,
                "type": "root",
                "children": [
                    {"concept": k, "count": v, "uri": f"{self.store.root_uri}/{k}"}
                    for k, v in sorted(counts.items())
                ],
            }

        concept = parts[0]

        if len(parts) == 1:
            entities = self.store.list_entities(concept)
            class_counts: dict[str, int] = {}
            for _c, cls, _obj, _u in entities:
                key = self.store.logical_class_name(concept, cls) or "(flat)"
                class_counts[key] = class_counts.get(key, 0) + 1
            return {
                "uri": uri,
                "type": "concept",
                "concept": concept,
                "children": [
                    {
                        "class_name": cls,
                        "count": cnt,
                        "uri": self.store.class_uri(concept, cls),
                    }
                    for cls, cnt in sorted(class_counts.items())
                ],
            }

        content = self.store.read_entity_by_uri(uri)
        if content is not None:
            info = self.store.lookup_by_uri(uri)
            c, cls, obj = info if info else (concept, None, parts[-1])
            result: dict[str, object] = {
                "uri": uri,
                "type": "entity",
                "concept": c,
                "class_name": self.store.logical_class_name(c, cls) or "",
                "object_name": obj,
                "path": str(
                    self.store.entity_path(c, cls, obj).relative_to(self.store.root),
                ),
                "content": content,
            }
            if c == "Symptom":
                result["children"] = self.list_symptom_profiles(uri)
            return result

        class_info = self.store.lookup_class_by_uri(uri)
        if class_info is not None:
            c, class_name = class_info
            logical_class_name = self.store.logical_class_name(c, class_name) or class_name
            entities = self.store.list_entities(c)
            if class_name == "(flat)":
                matched = [(cc, cls, obj, u) for cc, cls, obj, u in entities if not cls]
            else:
                matched = [
                    (cc, cls, obj, u)
                    for cc, cls, obj, u in entities
                    if (cls or "(flat)") == class_name or (cls or "").startswith(class_name + "/")
                ]
            return {
                "uri": uri,
                "type": "class",
                "concept": c,
                "class_name": logical_class_name,
                "children": [
                    {
                        "object_name": obj,
                        "uri": u,
                    }
                    for _cc, cls, obj, u in matched
                ],
            }

        return {"uri": uri, "type": "not_found", "error": f"No entities at {uri}"}

    def get_related_resources(
        self,
        uri: str,
        relation: str = "",
        cursor: int = 0,
        limit: int = 50,
    ) -> dict:
        """Return deterministic typed edges for one entity or Profile.

        This is the read side of the reference-maintenance worker.  It keeps
        structured frontmatter relations and narrative links separate, while
        carrying the source URIs that explain where each edge came from.
        ``cursor`` is an item offset, so callers can safely page a large
        component tree without changing the ordering between calls.
        """
        if cursor < 0:
            raise ValueError("cursor must be non-negative")
        if limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")

        content = self.read_resource(uri)
        if content is None:
            return {
                "uri": uri,
                "relation": relation,
                "cursor": cursor,
                "next_cursor": -1,
                "links": [],
                "error": "resource_not_found",
            }

        identity = self.store.lookup_by_uri(uri)
        concept = identity[0] if identity is not None else ""
        if "/profile/" in uri:
            concept = "SymptomProfile"
        frontmatter = parse_frontmatter(content)
        all_sources = list(dict.fromkeys(_source_values(frontmatter.get("sources"))))

        links: list[dict[str, object]] = []
        relation_fields = _RELATION_FIELDS.get(concept, ())
        for field in relation_fields:
            for target_uri in _uri_values(self.store.root_uri, frontmatter.get(field)):
                if target_uri == uri:
                    continue
                target_base = target_uri.partition("#")[0]
                target_concept = target_base.removeprefix(self.store.root_uri + "/").split("/", 1)[
                    0
                ]
                links.append(
                    {
                        "from_uri": uri,
                        "relation": field,
                        "to_uri": target_uri,
                        "to_concept": target_concept,
                        "source_refs": all_sources,
                        "target_section": "frontmatter",
                        "section_state": "complete",
                    },
                )

        for target_uri in sorted(extract_wiki_uris(content)):
            if target_uri == uri:
                continue
            section, state = _section_at(content, content.find(target_uri))
            target_base = target_uri.partition("#")[0]
            target_concept = target_base.removeprefix(self.store.root_uri + "/").split("/", 1)[0]
            links.append(
                {
                    "from_uri": uri,
                    "relation": "body_link",
                    "to_uri": target_uri,
                    "to_concept": target_concept,
                    "source_refs": all_sources,
                    "target_section": section,
                    "section_state": state,
                },
            )

        # OpenViking is the graph authority when the remote backend exposes
        # native relation edges.  Keep the Markdown-derived edges as the
        # explainable/source-bearing view, then add only native edges that are
        # not already represented.  This lets retrieval use the SDK without
        # losing section-level evidence required by the Wiki harness.
        native_relations = self.store._fs.relations(uri)
        known_native = {
            (str(link.get("relation", "")), str(link.get("to_uri", ""))) for link in links
        }
        for native in native_relations:
            to_uri = str(
                native.get("to_uri")
                or native.get("target_uri")
                or native.get("uri")
                or native.get("to", ""),
            ).strip()
            if not to_uri or to_uri == uri:
                continue
            native_relation = str(native.get("relation") or native.get("type") or "native_link")
            if (native_relation, to_uri) in known_native:
                continue
            known_native.add((native_relation, to_uri))
            links.append(
                {
                    "from_uri": uri,
                    "relation": native_relation,
                    "to_uri": to_uri,
                    "to_concept": to_uri.removeprefix(self.store.root_uri + "/").split("/", 1)[0],
                    "source_refs": [],
                    "target_section": "openviking_relation",
                    "section_state": "native",
                },
            )

        # A compatibility guard for legacy URI tokenizers: only retain links
        # that the canonical quality extractor can also see.  This prevents a
        # truncated ``/（）`` token from becoming a graph edge.
        canonical_links = extract_wiki_uris(content)
        links = [
            link
            for link in links
            if link.get("section_state") == "native" or link["to_uri"] in canonical_links
        ]
        links.sort(key=lambda link: (str(link["relation"]), str(link["to_uri"])))
        if relation:
            links = [link for link in links if link["relation"] == relation]
        page = links[cursor : cursor + limit]
        next_cursor = cursor + limit if cursor + limit < len(links) else -1
        return {
            "uri": uri,
            "relation": relation,
            "cursor": cursor,
            "next_cursor": next_cursor,
            "links": page,
        }

    def _list_bom_children(self, uri: str) -> dict:
        """Browse the global BOM tree (directories + markdown leaves).

        The configured BOM backend is a read-only namespace covering every
        machine class (e.g. ``viking://resources/<bom_namespace>/bom/component/挖掘机部件_BOM_清单``).
        Directories are returned as ``directory`` entries for drill-down;
        markdown files as ``bom`` entries whose per-machine model/tables can
        be read through ``read_resource``.
        """
        if self._bom_fs is None:
            return {"uri": uri, "type": "error", "error": "bom_root not configured"}
        prefix = self._bom_fs.root_uri.rstrip("/") + "/"
        key = uri.removeprefix(self._bom_fs.root_uri.rstrip("/")).strip("/")
        if not self._bom_fs.is_dir(key):
            return {
                "uri": uri,
                "type": "bom_file",
                "path": key,
            }
        entries: dict[str, str] = {}
        for rel in self._bom_fs.list_dir(key, recursive=True):
            tail = rel[len(key) :].lstrip("/") if key else rel
            if not tail:
                continue
            head, separator, _rest = tail.partition("/")
            entries[head] = "directory" if separator else "bom"
        return {
            "uri": uri,
            "type": "bom_dir",
            "children": [
                {
                    "name": head,
                    "type": kind,
                    "uri": f"{prefix}{key}/{head}" if key else f"{prefix}{head}",
                }
                for head, kind in sorted(entries.items())
            ],
        }

    def parent_of(self, uri: str) -> str | None:
        """Return the parent URI of a ``{root_uri}/`` URI, or ``None`` at root.

        Hierarchy:
        - ``{root_uri}/`` → ``None`` (already root)
        - ``{root_uri}/<Concept>`` → ``{root_uri}/``
        - ``{root_uri}/<Concept>/<class_hash>`` → ``{root_uri}/<Concept>``
        - ``{root_uri}/<Concept>/<entity_hash>`` → ``{root_uri}/<Concept>/<class_hash>``

        For entity URIs the class-hash parent is resolved via
        ``lookup_by_uri`` + ``class_uri``.  Non-wiki URIs return ``None``.
        """
        if not self.store.is_wiki_uri(uri):
            return None
        path = uri.removeprefix(self.store.root_uri).strip("/")
        parts = [p for p in path.split("/") if p]
        if not parts:
            return None
        if len(parts) == 1:
            return self.store.root_uri + "/"
        info = self.store.lookup_by_uri(uri)
        if info is not None:
            concept, class_name, _obj = info
            return (
                self.store.class_uri(concept, class_name)
                if class_name
                else f"{self.store.root_uri}/{concept}"
            )
        class_info = self.store.lookup_class_by_uri(uri)
        if class_info is not None:
            concept, _class_name = class_info
            return f"{self.store.root_uri}/{concept}"
        return None
