"""Checkpoint, link sync, and relation planning/closure."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import re
import time

from wolfharness.capabilities.wiki.quality import (
    BuildProfile,
    all_relation_uris,
    classify_raw_source_uri,
    extract_malformed_wiki_uris,
    extract_source_uris,
    has_unresolved_placeholder,
    is_optional_relation_field,
    parse_frontmatter,
)
from wolfharness.capabilities.wiki.schema_loader import get_concept_schema, get_schema_version
from wolfharness.capabilities.wiki.section_constants import (
    SECTION_COMMON_FAILURE_MODES,
    SECTION_DISASSEMBLY_STEPS,
)


logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING

from wolfharness.capabilities.wiki._helpers import (
    _RELATION_CLOSURE_READY_STAGES,
    _entity_batch_limit,
    _io_worker_limit,
)


if TYPE_CHECKING:
    from collections.abc import Callable


class RelationMixin:
    """Checkpoint, link sync, and relation planning/closure."""

    def checkpoint_build(
        self,
        doc_id: str,
        device_id: str,
        series_id: str,
        stage: str,
        *,
        build_id: str = "",
        input_hash: str = "",
        config_hash: str = "",
        source_snapshot_id: str = "",
        snapshot_id: str = "",
        input_docs: tuple[str, ...] = (),
        schema_version: str = "",
        audit_profile: BuildProfile = "manual",
        last_error_code: str = "",
        last_error: str = "",
    ) -> dict[str, object]:
        """Persist a resumable build checkpoint for the current wiki root."""
        checkpoint = self._build_checkpoint_record(
            doc_id,
            device_id,
            series_id,
            stage,
            build_id=build_id,
            input_hash=input_hash,
            config_hash=config_hash,
            source_snapshot_id=source_snapshot_id,
            snapshot_id=snapshot_id,
            input_docs=input_docs,
            schema_version=schema_version,
            audit_profile=audit_profile,
            last_error_code=last_error_code,
            last_error=last_error,
        )
        self.store.write_json("index/build_checkpoint.json", checkpoint)
        if stage in _RELATION_CLOSURE_READY_STAGES:
            self._relation_manifest_packet_ids(checkpoint)
        return {"exists": True, **checkpoint}

    def _build_checkpoint_record(
        self,
        doc_id: str,
        device_id: str,
        series_id: str,
        stage: str,
        *,
        build_id: str = "",
        input_hash: str = "",
        config_hash: str = "",
        source_snapshot_id: str = "",
        snapshot_id: str = "",
        input_docs: tuple[str, ...] = (),
        schema_version: str = "",
        audit_profile: BuildProfile = "manual",
        last_error_code: str = "",
        last_error: str = "",
    ) -> dict[str, object]:
        """Build checkpoint content without changing the visible build state."""
        if not stage.strip():
            raise ValueError("checkpoint stage must not be empty")
        audit_profile = self._validate_audit_profile(audit_profile)
        schema_version = schema_version or get_schema_version()
        existing = self.store.read_json("index/build_checkpoint.json")
        if not build_id:
            existing_build_id = (
                str(existing.get("build_id", "")) if isinstance(existing, dict) else ""
            )
            existing_doc_id = str(existing.get("doc_id", "")) if isinstance(existing, dict) else ""
            if existing_doc_id == doc_id and existing_build_id:
                # Preserve the existing build identity.  A checkpoint_build
                # call without an explicit build_id (recovery paths, stage
                # transitions) must not mint a fresh auto-generated id over an
                # in-progress build — that silently detaches every plan and
                # source packet owner, leaving materialized chapters pending
                # forever.  Only brand-new builds (no matching checkpoint)
                # auto-generate an id.
                build_id = existing_build_id
            else:
                build_id = sha256(
                    f"{doc_id}\x1f{device_id}\x1f{series_id}\x1f{input_hash}\x1f{config_hash}\x1f{','.join(input_docs)}\x1f{schema_version}\x1f{audit_profile}".encode(),
                ).hexdigest()[:16]
        existing_build_id = str(existing.get("build_id", "")) if isinstance(existing, dict) else ""
        if existing is not None:
            existing_profile = str(existing.get("audit_profile", "manual"))
            if build_id == existing_build_id and existing_profile != audit_profile:
                raise ValueError(
                    f"Build audit profile cannot change after checkpoint creation: existing={existing_profile}, requested={audit_profile}",
                )
        started_at = (
            str(existing.get("started_at", ""))
            if isinstance(existing, dict) and existing_build_id == build_id
            else ""
        )
        if not started_at:
            started_at = datetime.now(UTC).isoformat()
        updated_at = datetime.now(UTC).isoformat()
        # Code-level source identity: combines library_root + source_doc_allowlist
        # + wiki_root into a deterministic fingerprint.  inspect_build_checkpoint
        # validates this to detect source changes (e.g. fixmaster → local
        # OpenViking, or different namespace) without relying on LLM-passed
        # doc_id/build_id.
        _source_fp = sha256(
            f"{self._raw_fs.root_uri}\x1f{','.join(getattr(self, '_source_doc_allowlist', ()))}\x1f{self.store.root_uri}".encode(),
        ).hexdigest()[:16]
        return {
            "build_id": build_id,
            "doc_id": doc_id,
            "device_id": device_id,
            "series_id": series_id,
            "stage": stage,
            "input_hash": input_hash,
            "config_hash": config_hash,
            "source_snapshot_id": source_snapshot_id,
            "snapshot_id": snapshot_id,
            # ponytail: conductor prompt omits input_docs → default to doc_id
            # so packets.py has_input_identity check passes.  Matches fallback
            # already present in chapters.py:904 and materialization.py:235.
            "input_docs": list(input_docs) if input_docs else [doc_id],
            "schema_version": schema_version,
            "audit_profile": audit_profile,
            "source_fingerprint": _source_fp,
            "library_root": self._raw_fs.root_uri,
            "last_error_code": last_error_code,
            "last_error": last_error,
            "started_at": started_at,
            "updated_at": updated_at,
        }

    # ── Entity CRUD ─────────────────────────────────────────────────────────

    def _related_entity_links(
        self,
        concept: str,
        relation_field: str,
        component_uri: str,
    ) -> list[tuple[str, str]]:
        """Find real target pages whose relation field names *component_uri*."""
        links: list[tuple[str, str]] = []
        for _concept, _class_name, object_name, uri in self.store.list_entities(concept):
            page = self.store.read_entity_by_uri(uri)
            if page is None:
                continue
            relation_uris = all_relation_uris(page, concept, relation_field, self.store.root_uri)
            if component_uri not in relation_uris:
                continue
            frontmatter = parse_frontmatter(page)
            title = frontmatter.get("title")
            label = title.strip() if isinstance(title, str) and title.strip() else object_name
            links.append((label, uri))
        return links

    def _sync_component_narrative_links(
        self,
        component_uri: str,
        *,
        fault_links: list[tuple[str, str]] | None = None,
        procedure_links: list[tuple[str, str]] | None = None,
        force: bool = False,
    ) -> bool:
        """Materialize reverse Fault/Procedure edges in a Component body.

        Structured edges remain authoritative in Fault/Procedure frontmatter;
        this helper only mirrors those already-resolved edges into the two
        human-readable Component sections required by the graph contract.
        Returns True when the Component body was rewritten.
        """
        with self._relation_sync_lock:
            return self._sync_component_narrative_links_locked(
                component_uri,
                fault_links=fault_links,
                procedure_links=procedure_links,
                force=force,
            )

    def _sync_component_narrative_links_locked(
        self,
        component_uri: str,
        *,
        fault_links: list[tuple[str, str]] | None = None,
        procedure_links: list[tuple[str, str]] | None = None,
        force: bool = False,
    ) -> bool:
        """Run Component reverse-link synchronization under its shared lock.

        Returns True when the Component body was rewritten.
        """
        relation_mode = os.environ.get("WIKI_RELATION_SYNC_MODE", "immediate").strip().lower()
        if not force and relation_mode == "deferred":
            return False
        info = self.store.lookup_by_uri(component_uri)
        if info is None or info[0] != "Component":
            return False
        concept, class_name, object_name = info
        content = self.store.read_entity_by_uri(component_uri)
        if content is None:
            return False
        resolved_fault_links = (
            fault_links
            if fault_links is not None
            else self._related_entity_links("Fault", "affected_components", component_uri)
        )
        resolved_procedure_links = (
            procedure_links
            if procedure_links is not None
            else self._related_entity_links("Procedure", "target_components", component_uri)
        )
        updated = self._append_section_links(
            content, SECTION_COMMON_FAILURE_MODES, resolved_fault_links
        )
        updated = self._append_section_links(
            updated, SECTION_DISASSEMBLY_STEPS, resolved_procedure_links
        )
        updated = self._dedupe_h2_sections(updated)
        if updated == content:
            return False
        updated = self.store.resolve_body_refs(updated, None)
        updated = self.store.dedup_citations(updated)
        updated = self._preserve_expert_sections(
            target_uri=component_uri,
            current=content,
            candidate=updated,
        )
        if updated == content:
            return False
        self.store.write_entity(concept, class_name, object_name, updated)
        self.store.register_natural_key(concept, class_name, object_name, component_uri)
        logger.info(
            "Materialized Component narrative links: %s (faults=%d, procedures=%d)",
            component_uri,
            len(resolved_fault_links),
            len(resolved_procedure_links),
        )
        return True

    def _sync_all_component_narrative_links(self, *, force: bool = False) -> int:
        """Mirror reverse Fault/Procedure links into every Component body.

        The join step after relation shards finish.  Builds a reverse index in
        one pass over Fault/Procedure pages instead of re-scanning all entities
        once per component (O(C*N) reads -> O(N) reads), then syncs only the
        Components that actually have reverse links.  Returns how many
        Component bodies changed.
        """
        fault_index: dict[str, list[tuple[str, str]]] = {}
        for _concept, _class_name, object_name, uri in self.store.list_entities("Fault"):
            page = self.store.read_entity_by_uri(uri)
            if page is None:
                continue
            component_uris = all_relation_uris(
                page, "Fault", "affected_components", self.store.root_uri
            )
            if not component_uris:
                continue
            frontmatter = parse_frontmatter(page)
            title = frontmatter.get("title")
            label = title.strip() if isinstance(title, str) and title.strip() else object_name
            for component_uri in component_uris:
                fault_index.setdefault(component_uri, []).append((label, uri))
        procedure_index: dict[str, list[tuple[str, str]]] = {}
        for _concept, _class_name, object_name, uri in self.store.list_entities("Procedure"):
            page = self.store.read_entity_by_uri(uri)
            if page is None:
                continue
            component_uris = all_relation_uris(
                page, "Procedure", "target_components", self.store.root_uri
            )
            if not component_uris:
                continue
            frontmatter = parse_frontmatter(page)
            title = frontmatter.get("title")
            label = title.strip() if isinstance(title, str) and title.strip() else object_name
            for component_uri in component_uris:
                procedure_index.setdefault(component_uri, []).append((label, uri))
        synced = 0
        for component_uri in sorted(set(fault_index) | set(procedure_index)):
            # Join is the deferred-mode executor: always mirror, regardless of
            # WIKI_RELATION_SYNC_MODE (workers may have skipped immediate sync).
            if self._sync_component_narrative_links(
                component_uri,
                fault_links=fault_index.get(component_uri),
                procedure_links=procedure_index.get(component_uri),
                force=True,
            ):
                synced += 1
        return synced

    def _sync_symptom_profile_index(self, symptom_uri: str, *, force: bool = False) -> None:
        """Backfill ## Profile 索引 in a Symptom index.md from its profile files."""
        with self._relation_sync_lock:
            self._sync_symptom_profile_index_locked(symptom_uri, force=force)

    def _sync_symptom_profile_index_locked(self, symptom_uri: str, *, force: bool = False) -> None:
        """Run Symptom profile index synchronization under its shared lock."""
        relation_mode = os.environ.get("WIKI_RELATION_SYNC_MODE", "immediate").strip().lower()
        if not force and relation_mode == "deferred":
            return
        info = self.store.lookup_by_uri(symptom_uri)
        if info is None or info[0] != "Symptom":
            return
        concept, class_name, object_name = info
        content = self.store.read_entity_by_uri(symptom_uri)
        if content is None:
            return
        profiles = self.store.list_symptom_profiles(symptom_uri)
        if not profiles:
            return
        rows: list[str] = []
        for profile_id, profile_uri in profiles:
            profile_content = self.store.read_entity_by_uri(profile_uri)
            if profile_content is None:
                continue
            fm = parse_frontmatter(profile_content)
            device_ref = fm.get("device_refs", [])
            direct_comp = fm.get("direct_component_uri", "")
            applicable = fm.get("applicable_models", [])
            if isinstance(applicable, list) and applicable:
                desc = ", ".join(str(item) for item in applicable[:2])
            elif isinstance(device_ref, list) and device_ref:
                first = device_ref[0]
                desc = first.rsplit("/", 1)[-1] if isinstance(first, str) else profile_id
            else:
                desc = profile_id
            if direct_comp and isinstance(direct_comp, str):
                desc = f"{desc} + {direct_comp.rsplit('/', 1)[-1]}"
            rows.append(f"| [{profile_id}]({profile_uri}) | {desc} |")
        if not rows:
            return
        table_header = "| Profile | 设备 + 直接关联 Component |\n|---|---|"
        table_body = "\n".join(rows)
        section_body = f"{table_header}\n{table_body}"
        updated = self._replace_h2_section(content, "Profile 索引", section_body)
        updated = self._dedupe_h2_sections(updated)
        if updated == content:
            return
        updated = self.store.resolve_body_refs(updated, None)
        updated = self.store.dedup_citations(updated)
        updated = self._preserve_expert_sections(
            target_uri=symptom_uri,
            current=content,
            candidate=updated,
        )
        if updated == content:
            return
        self.store.write_entity(concept, class_name, object_name, updated)
        self.store.register_natural_key(concept, class_name, object_name, symptom_uri)
        logger.info("Synced Symptom profile index: %s (%d profiles)", symptom_uri, len(rows))

    @staticmethod
    def _replace_h2_section(content: str, heading: str, body: str) -> str:
        """Replace one existing ``##`` section or append it once."""
        matches = list(re.finditer(r"^##\s+(.+?)\s*$", content, re.MULTILINE))
        target_index = next(
            (index for index, match in enumerate(matches) if match.group(1).strip() == heading),
            None,
        )
        rendered = f"\n\n## {heading}\n\n{body.rstrip()}\n"
        if target_index is None:
            return content.rstrip() + rendered
        match = matches[target_index]
        end = matches[target_index + 1].start() if target_index + 1 < len(matches) else len(content)
        return (
            content[: match.start()]
            + f"## {heading}\n\n{body.rstrip()}\n\n"
            + content[end:].lstrip("\n")
        )

    def _sync_device_diagnostic_links(self) -> int:
        """Materialize each Device's resolved Symptom→Fault→Component table.

        Device ``symptom_refs`` and Profile/Fault frontmatter are the
        authoritative graph.  This pass only mirrors already-resolved URIs;
        it never invents a relation from display names.  It runs once at the
        relation join, so workers remain isolated while the Device page gets a
        deterministic, complete diagnostic index.
        """
        synced = 0
        with self._relation_sync_lock:
            # Pre-collect DTCs by model prefix for the 控制器与故障码 section.
            # DTC class_name format: <model>_<controller_role> (e.g. SY75C_主控制器)
            dtc_by_model: dict[str, list[tuple[str, str, str, str]]] = {}
            for _c, dtc_class, _o, dtc_uri in self.store.list_entities("DTC"):
                dtc_content = self.store.read_entity_by_uri(dtc_uri)
                if dtc_content is None:
                    continue
                dtc_fm = parse_frontmatter(dtc_content)
                code = str(dtc_fm.get("code", _o))
                role = str(dtc_fm.get("controller_role", ""))
                if not role and "_" in dtc_class:
                    role = dtc_class.split("_", 1)[-1]
                fault_uris = list(
                    all_relation_uris(dtc_content, "DTC", "related_faults", self.store.root_uri)
                )
                fault_display = ""
                if fault_uris:
                    fc = self.store.read_entity_by_uri(fault_uris[0])
                    if fc:
                        ff = parse_frontmatter(fc)
                        fname = str(ff.get("title", fault_uris[0].rsplit("/", 1)[-1]))
                        fault_display = f"[{fname.replace('|', '／')}]({fault_uris[0]})"
                model_prefix = dtc_class.split("_", 1)[0] if "_" in dtc_class else dtc_class
                dtc_by_model.setdefault(model_prefix, []).append((
                    role,
                    code,
                    dtc_uri,
                    fault_display,
                ))
            for _concept, class_name, object_name, device_uri in self.store.list_entities("Device"):
                content = self.store.read_entity_by_uri(device_uri)
                if content is None:
                    continue
                device_components = all_relation_uris(
                    content, "Device", "critical_components", self.store.root_uri
                )
                symptom_uris = all_relation_uris(
                    content, "Device", "symptom_refs", self.store.root_uri
                )
                device_fm = parse_frontmatter(content)
                raw_models = device_fm.get("applicable_models", [])
                if isinstance(raw_models, list):
                    device_models = [str(m) for m in raw_models]
                elif raw_models:
                    device_models = [str(raw_models)]
                else:
                    device_models = []
                dtc_rows: list[tuple[str, str, str, str]] = []
                for model in device_models:
                    dtc_rows.extend(dtc_by_model.get(model, []))
                rows: dict[tuple[str, str, str], tuple[str, str, str, str]] = {}
                symptom_links: list[tuple[str, str]] = []

                for symptom_uri in symptom_uris:
                    symptom_content = self.store.read_entity_by_uri(symptom_uri)
                    if symptom_content is None:
                        continue
                    symptom_frontmatter = parse_frontmatter(symptom_content)
                    symptom_name = str(
                        symptom_frontmatter.get("title", symptom_uri.rsplit("/", 1)[-1])
                    )
                    symptom_links.append((symptom_name, symptom_uri))
                    symptom_sources = extract_source_uris(symptom_content)
                    profiles = self.list_symptom_profiles(symptom_uri)
                    for profile in profiles:
                        profile_uri = str(profile.get("uri", ""))
                        profile_content = self.store.read_entity_by_uri(profile_uri)
                        if profile_content is None:
                            continue
                        profile_devices = all_relation_uris(
                            profile_content, "Symptom", "device_refs", self.store.root_uri
                        )
                        if profile_devices and device_uri not in profile_devices:
                            continue
                        direct_component = list(
                            all_relation_uris(
                                profile_content,
                                "Symptom",
                                "direct_component_uri",
                                self.store.root_uri,
                            )
                        )
                        profile_faults = list(
                            all_relation_uris(
                                profile_content,
                                "Symptom",
                                "possible_faults",
                                self.store.root_uri,
                            )
                        )
                        for fault_uri in profile_faults:
                            fault_content = self.store.read_entity_by_uri(fault_uri)
                            if fault_content is None:
                                continue
                            fault_frontmatter = parse_frontmatter(fault_content)
                            affected_components = list(
                                all_relation_uris(
                                    fault_content,
                                    "Fault",
                                    "affected_components",
                                    self.store.root_uri,
                                )
                            )
                            component_candidates = [
                                uri
                                for uri in dict.fromkeys([*direct_component, *affected_components])
                                if not device_components or uri in device_components
                            ]
                            if not component_candidates:
                                continue
                            fault_name = str(
                                fault_frontmatter.get("title", fault_uri.rsplit("/", 1)[-1])
                            )
                            raw_sources = [
                                uri
                                for uri in [
                                    *extract_source_uris(profile_content),
                                    *extract_source_uris(fault_content),
                                    *symptom_sources,
                                ]
                                if classify_raw_source_uri(uri, raw_root_uri=self._raw_fs.root_uri)
                                is not None
                            ]
                            evidence = raw_sources[0] if raw_sources else ""
                            for component_uri in component_candidates:
                                component_info = self.store.lookup_by_uri(component_uri)
                                component_name = (
                                    str(component_info[2])
                                    if component_info is not None
                                    else component_uri.rsplit("/", 1)[-1]
                                )
                                key = (symptom_uri, fault_uri, component_uri)
                                rows[key] = (symptom_name, fault_name, component_name, evidence)

                if not rows:
                    if not symptom_links and not dtc_rows:
                        continue
                    # No profile→fault→component chain resolved; fall back to
                    # listing symptom_refs as readable links in the body.
                    updated = content
                    if symptom_links:
                        table_lines = [
                            "| 故障现象 |",
                            "| --- |",
                        ]
                        for sname, suri in sorted(symptom_links):
                            table_lines.append(f"| [{sname.replace('|', '／')}]({suri}) |")
                        updated = self._replace_h2_section(
                            updated, "常见故障及故障机理", "\n".join(table_lines)
                        )
                        updated = self._dedupe_h2_sections(updated)
                        updated = self.store.resolve_body_refs(updated, None)
                        updated = self.store.dedup_citations(updated)
                    # DTC section
                    if dtc_rows:
                        dtc_table = [
                            "| 控制器 | 故障码 | 关联失效 |",
                            "| --- | --- | --- |",
                        ]
                        for role, code, dtc_uri, fault_display in sorted(dtc_rows):
                            dtc_table.append(
                                f"| {role} | [{code}]({dtc_uri}) | {fault_display or '—'} |"
                            )
                        updated = self._replace_h2_section(
                            updated, "控制器与故障码", "\n".join(dtc_table)
                        )
                        updated = self._dedupe_h2_sections(updated)
                    if updated != content:
                        updated = self._preserve_expert_sections(
                            target_uri=device_uri,
                            current=content,
                            candidate=updated,
                        )
                    if updated != content:
                        self.store.write_entity("Device", class_name, object_name, updated)
                        synced += 1
                    continue
                table_lines = [
                    "| 故障现象 | 失效机理 | 关联部件 | 原文证据 |",
                    "| --- | --- | --- | --- |",
                ]
                for symptom_uri, fault_uri, component_uri in sorted(rows):
                    symptom_name, fault_name, component_name, evidence = rows[
                        (symptom_uri, fault_uri, component_uri)
                    ]
                    evidence_cell = f"[原文]({evidence})" if evidence else "来源未说明"
                    table_lines.append(
                        "| "
                        f"[{symptom_name.replace('|', '／')}]({symptom_uri}) | "
                        f"[{fault_name.replace('|', '／')}]({fault_uri}) | "
                        f"[{component_name.replace('|', '／')}]({component_uri}) | {evidence_cell} |",
                    )
                updated = self._replace_h2_section(
                    content, "常见故障及故障机理", "\n".join(table_lines)
                )
                updated = self._dedupe_h2_sections(updated)
                updated = self.store.resolve_body_refs(updated, None)
                updated = self.store.dedup_citations(updated)
                # DTC section
                if dtc_rows:
                    dtc_table = [
                        "| 控制器 | 故障码 | 关联失效 |",
                        "| --- | --- | --- |",
                    ]
                    for role, code, dtc_uri, fault_display in sorted(dtc_rows):
                        dtc_table.append(
                            f"| {role} | [{code}]({dtc_uri}) | {fault_display or '—'} |"
                        )
                    updated = self._replace_h2_section(
                        updated, "控制器与故障码", "\n".join(dtc_table)
                    )
                    updated = self._dedupe_h2_sections(updated)
                if updated != content:
                    updated = self._preserve_expert_sections(
                        target_uri=device_uri,
                        current=content,
                        candidate=updated,
                    )
                if updated != content:
                    self.store.write_entity("Device", class_name, object_name, updated)
                    synced += 1
        return synced

    def _relation_field_shape(self, concept: str, field: str) -> str:
        """Return the schema-declared shape of one relation field."""
        schema = get_concept_schema(concept)
        frontmatter = schema.get("frontmatter")
        if not isinstance(frontmatter, list):
            return "unknown"
        for raw_field in frontmatter:
            if not isinstance(raw_field, dict) or raw_field.get("name") != field:
                continue
            field_type = raw_field.get("type")
            return str(field_type) if isinstance(field_type, str) else "unknown"
        return "unknown"

    def _relation_field_names(self, concept: str) -> tuple[str, ...]:
        """Return relation fields from the authoritative schema."""
        schema = get_concept_schema(concept)
        frontmatter = schema.get("frontmatter")
        if not isinstance(frontmatter, list):
            return ()
        names = [
            str(item["name"])
            for item in frontmatter
            if isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and item.get("type") in {"ref", "ref_list"}
        ]
        return tuple(names)

    def _entity_relation_work_items(
        self,
        *,
        scope_entity_uris: list[str] | None = None,
    ) -> list[dict[str, object]]:
        """Discover relation work from committed entities, not extraction state.

        Source packets are evidence for extraction and checkpoints are restart
        metadata. Neither is authoritative for deciding whether a committed
        page still needs a relation. This scan looks only at the current
        entity library, schema relation fields and unresolved placeholders;
        OPA/OPS review state is deliberately not a relation-work trigger.

        When ``scope_entity_uris`` is provided, the scan is limited to those
        entities only — used in incremental mode to avoid re-planning relation
        work for old entities untouched by the current build.
        """
        all_entity_records = self.store.list_entities()
        known_uris = {str(record[3]) for record in all_entity_records}
        if scope_entity_uris is not None:
            scope_set = {uri.strip() for uri in scope_entity_uris if uri.strip()}
            entity_records = [r for r in all_entity_records if str(r[3]) in scope_set]
        else:
            entity_records = all_entity_records
        root_uri = self.store.root_uri.rstrip("/") + "/"

        def relation_values(value: object) -> list[str]:
            if isinstance(value, str):
                return [value.strip()] if value.strip().startswith(root_uri) else []
            if isinstance(value, list):
                return [uri for item in value for uri in relation_values(item)]
            return []

        items: list[dict[str, object]] = []
        for concept, class_name, object_name, uri in entity_records:
            entity_uri = str(uri)
            content = self.store.read_entity_by_uri(entity_uri)
            if content is None:
                continue
            relation_fields = tuple(
                field
                for field in self._relation_field_names(str(concept))
                if not is_optional_relation_field(str(concept), field)
            )
            if not relation_fields:
                # Under the default backbone policy this page has no
                # publication-critical outgoing relation. Do not schedule an
                # LLM merely to beautify optional Procedure/DTC/Part links.
                continue
            frontmatter = parse_frontmatter(content)
            missing_fields: list[str] = []
            dangling_fields: list[str] = []
            for field in relation_fields:
                value = frontmatter.get(field)
                values = relation_values(value)
                if value is None or value in ("", []):
                    # Body fallback: check if the body section has URIs before
                    # flagging this field missing (body-first authoring).
                    if all_relation_uris(content, str(concept), field, root_uri):
                        continue
                    missing_fields.append(field)
                    continue
                if (
                    values
                    and any(uri_value.split("#", 1)[0] not in known_uris for uri_value in values)
                ) or (isinstance(value, str) and value.strip().startswith(root_uri) and not values):
                    dangling_fields.append(field)

            placeholder = bool(has_unresolved_placeholder(content))
            malformed = bool(extract_malformed_wiki_uris(content))
            reasons: list[str] = []
            if missing_fields:
                reasons.append("missing_relation_field")
            if dangling_fields:
                reasons.append("dangling_relation_uri")
            if placeholder:
                reasons.append("unresolved_placeholder")
            if malformed:
                reasons.append("malformed_uri")
            if not reasons:
                continue
            query_parts = [str(object_name), str(class_name)]
            query_parts.extend(missing_fields)
            query_parts.extend(dangling_fields)
            query = " ".join(part for part in query_parts if part.strip())
            items.append(
                {
                    "entity_uri": entity_uri,
                    "concept": str(concept),
                    "class_name": str(class_name),
                    "object_name": str(object_name),
                    "write_set": [entity_uri],
                    "relation_fields": sorted({*missing_fields, *dangling_fields}),
                    "reasons": reasons,
                    "retrieval_query": query,
                    "expected_sha256": sha256(content.encode("utf-8")).hexdigest(),
                },
            )
        return items

    def plan_relation_work(
        self,
        *,
        active_entity_uris: list[str] | None = None,
        max_parallel_shards: int | None = None,
        scope_entity_uris: list[str] | None = None,
    ) -> dict[str, object]:
        """Plan retrieval-and-patch work from the current committed Wiki.

        This is the primary relation planner. It intentionally does not read
        checkpoints, manifests or source packets. A returned item is a bounded
        entity write set; the worker must retrieve real neighbouring pages,
        verify identity/evidence, then use guarded ``patch_entity``. Empty
        fields are review work, never permission to invent a URI.

        When ``scope_entity_uris`` is ``None`` and the wiki is in incremental
        mode, the planner auto-scopes to entities touched in the current build
        via ``build_change_report``. Pass an explicit list to override; pass
        ``None`` in a non-incremental wiki for the traditional full-library scan.
        """
        if max_parallel_shards is not None and max_parallel_shards < 1:
            raise ValueError("max_parallel_shards must be positive when provided")
        active_uris = {uri.strip() for uri in active_entity_uris or [] if uri.strip()}

        # Auto-scope in incremental mode: limit to build-touched entities
        auto_scoped = False
        if scope_entity_uris is None:
            wiki_state = self.inspect_wiki_state()
            if wiki_state.get("mode") == "incremental":
                try:
                    change_report = self.build_change_report(persist=False, include_op_flow=False)
                    changed = change_report.get("changed_entities", [])
                    if isinstance(changed, list):
                        scope_entity_uris = [
                            str(e.get("uri", "")).strip()
                            for e in changed
                            if isinstance(e, dict) and e.get("uri")
                        ]
                        auto_scoped = True
                except Exception:
                    # If build_change_report fails, fall back to full library scan
                    pass

        relation_items = self._entity_relation_work_items(scope_entity_uris=scope_entity_uris)
        grouped: dict[str, list[dict[str, object]]] = {}
        for item in relation_items:
            grouped.setdefault(str(item["concept"]), []).append(item)
        all_shards: list[dict[str, object]] = []
        active_excluded_count = 0
        shard_index = 0
        for concept in sorted(grouped):
            concept_items = grouped[concept]
            for start in range(0, len(concept_items), _entity_batch_limit()):
                shard_index += 1
                chunk = concept_items[start : start + _entity_batch_limit()]
                entity_uris = [str(item["entity_uri"]) for item in chunk]
                shard_id = f"entity_relation_{concept.casefold()}_{shard_index}"
                task_description = "\n".join(
                    [
                        "worker_role: wiki_relation_worker",
                        "depends_on_stage: entity_library_materialized",
                        "relation_planner: plan_relation_work",
                        f"relation_shard_id: {shard_id}",
                        f"write_scope: entity_relation_retrieval:{shard_id}",
                        f"write_set: {json.dumps(entity_uris, ensure_ascii=False, separators=(',', ':'))}",
                        "relation_work_items: "
                        + json.dumps(chunk, ensure_ascii=False, separators=(",", ":")),
                        "expected_artifacts: patched_entities_or_source_honest_open_gap",
                    ],
                )
                shard: dict[str, object] = {
                    "shard_id": shard_id,
                    "concept": concept,
                    "entity_uris": entity_uris,
                    "write_set": entity_uris,
                    "relation_work_items": chunk,
                    "candidate_count": len(chunk),
                    "write_scope": f"entity_relation_retrieval:{shard_id}",
                    "task_description": task_description,
                }
                if active_uris.intersection(entity_uris):
                    active_excluded_count += len(chunk)
                    continue
                all_shards.append(shard)
        shards = all_shards[:max_parallel_shards] if max_parallel_shards is not None else all_shards
        returned_candidate_count = sum(int(shard["candidate_count"]) for shard in shards)
        dispatchable_candidate_count = sum(int(shard["candidate_count"]) for shard in all_shards)
        current_uris = sorted(
            known_uri for known_uri in {str(record[3]) for record in self.store.list_entities()}
        )
        if scope_entity_uris is not None:
            scope_uris_for_id = sorted(set(scope_entity_uris))
        else:
            scope_uris_for_id = current_uris
        scope_id = (
            "entity-relation-"
            + sha256("\n".join(scope_uris_for_id).encode("utf-8")).hexdigest()[:16]
        )
        effective_scope_source = (
            "build_touched_entities" if scope_entity_uris is not None else "entity_library"
        )
        ledger_uri = self._write_relation_work_ledger(
            scope_id,
            scope_source=effective_scope_source,
            scope_trusted=True,
            packet_ids=set(),
            items=relation_items,
        )
        return {
            "scope_source": effective_scope_source,
            "scope_id": scope_id,
            "work_ledger_uri": ledger_uri,
            "shards": shards,
            "shard_count": len(shards),
            "candidate_count": returned_candidate_count,
            "pending_count": len(relation_items),
            "active_excluded_count": active_excluded_count,
            "remaining_count": dispatchable_candidate_count - returned_candidate_count,
            "scope_entity_count": len(scope_entity_uris) if scope_entity_uris is not None else None,
            "auto_scoped": auto_scoped,
            "join": {
                "required": False,
                "service_action": "rebuild_all_backlinks",
                "entity_uris": sorted({str(item["entity_uri"]) for item in relation_items}),
                "write_set": sorted({str(item["entity_uri"]) for item in relation_items}),
                "sync_component_links": True,
                "depends_on_shards": [str(shard["shard_id"]) for shard in shards],
                "task_description": "",
            },
            "idempotent": True,
        }

    def _resolve_relation_identity(self, value: object) -> str | None:
        """Resolve a packet identity without fuzzy name matching."""
        if not isinstance(value, dict):
            return None
        concept = value.get("concept")
        class_name = value.get("class_name", value.get("class"))
        object_name = value.get("object_name", value.get("object"))
        if not all(isinstance(item, str) and item.strip() for item in (concept, object_name)):
            return None
        assert isinstance(concept, str)
        assert isinstance(object_name, str)
        normalized_class = (
            class_name.strip() if isinstance(class_name, str) and class_name.strip() else None
        )
        return self.store.lookup_natural_key(concept, normalized_class, object_name.strip())

    def _resolve_relation_uri(self, value: object) -> str | None:
        """Resolve an explicitly supplied URI only when its page exists."""
        if not isinstance(value, str) or not value.strip():
            return None
        candidate = value.strip()
        if candidate in {"open_gap", "needs_predecessor", "unresolved"}:
            return None
        resolved = self.store.resolve_redirect(candidate)
        if self.store.lookup_by_uri(resolved) is None:
            return None
        return resolved if self.store.read_entity_by_uri(resolved) is not None else None

    def _relation_context(
        self,
        selected_packets: set[str],
    ) -> tuple[
        dict[str, tuple[str, str, str]],
        Callable[[object], str | None],
        Callable[[object], str | None],
        list[tuple[str, dict[str, object], str]],
    ]:
        """Load one immutable relation snapshot for planning and closure.

        Relation planning and execution must resolve identities from the same
        snapshot.  Keeping this discovery in one helper prevents the planner
        from drifting away from the actual closure semantics and avoids a
        remote lookup for every candidate endpoint.
        """
        entity_records = self.store.list_entities()
        known_uris = {str(record[3]) for record in entity_records}
        known_info_by_uri = {
            str(record[3]): (str(record[0]), str(record[1]), str(record[2]))
            for record in entity_records
        }
        known_by_identity = {
            (
                str(record[0]),
                str(self.store.logical_class_name(record[0], record[1]) or ""),
                str(record[2]),
            ): str(record[3])
            for record in entity_records
        }

        def resolve_uri(value: object) -> str | None:
            if not isinstance(value, str) or not value.strip():
                return None
            candidate = value.strip().split("#", 1)[0]
            if candidate in known_uris:
                return candidate
            return self._resolve_relation_uri(candidate)

        def resolve_identity(value: object) -> str | None:
            if not isinstance(value, dict):
                return None
            concept = value.get("concept")
            class_name = value.get("class_name", "")
            object_name = value.get("object_name")
            if not all(isinstance(item, str) for item in (concept, class_name, object_name)):
                return None
            key = (
                concept,
                str(self.store.logical_class_name(concept, class_name) or ""),
                object_name,
            )
            return known_by_identity.get(key) or self._resolve_relation_identity(value)

        packet_keys = [
            key
            for key in self.store.list_dir("source_packets", recursive=True)
            if key.endswith(".json")
        ]
        if selected_packets:
            packet_keys = [key for key in packet_keys if Path(key).stem in selected_packets]

        def read_packet(key: str) -> tuple[str, dict[str, object] | None]:
            return key, self.store.read_json(key)

        with ThreadPoolExecutor(
            max_workers=min(_io_worker_limit(), max(1, len(packet_keys)))
        ) as pool:
            packet_records = list(pool.map(read_packet, packet_keys))
        candidates: list[tuple[str, dict[str, object], str]] = []
        for key, packet in packet_records:
            if not isinstance(packet, dict):
                continue
            packet_id = str(packet.get("packet_id", ""))
            if selected_packets and packet_id not in selected_packets:
                continue
            body = packet.get("packet")
            if not isinstance(body, dict):
                continue
            raw_edges: object = body.get("local_relation_edges")
            if raw_edges is None:
                raw_edges = body.get("cross_concept_handoffs")
            if raw_edges is None:
                raw_edges = body.get("relation_candidates")
            if isinstance(raw_edges, list):
                candidates.extend(
                    (packet_id, raw_edge, key)
                    for raw_edge in raw_edges
                    if isinstance(raw_edge, dict)
                )
        return known_info_by_uri, resolve_uri, resolve_identity, candidates

    def plan_relation_shards(
        self,
        entity_uris: list[str] | None = None,
        *,
        packet_ids: list[str] | None = None,
    ) -> dict[str, object]:
        """Plan only the non-empty, disjoint relation write shards.

        Shards are derived from materialized relation candidates, not from a
        fixed worker count.  A source entity belongs to exactly one concept
        shard, so two shard workers never patch the same source URI.  Optional
        relation fields and unresolved source-honest gaps are reported but do
        not create empty tasks.
        """
        checkpoint = self.inspect_build_checkpoint()
        stage = str(checkpoint.get("stage", ""))
        if not checkpoint.get("exists") or stage not in _RELATION_CLOSURE_READY_STAGES:
            current_stage = stage or "missing"
            raise ValueError(
                f"Relation planning is gated until materialization is complete; current checkpoint stage={current_stage!r}. Call checkpoint_build(stage='materialized') first.",
            )

        global_scope = entity_uris is None and packet_ids is None
        manifest_packet_ids = self._relation_manifest_packet_ids(checkpoint)
        scope = {
            uri.split("#", 1)[0]
            for uri in (entity_uris or [])
            if isinstance(uri, str) and uri.strip()
        }
        selected_packets = {
            packet_id.strip() for packet_id in (packet_ids or []) if packet_id.strip()
        }
        if packet_ids is None:
            selected_packets = set(manifest_packet_ids)
        known_info_by_uri, resolve_uri, resolve_identity, candidates = self._relation_context(
            selected_packets
        )
        groups: dict[str, dict[str, object]] = {}
        unresolved_count = 0
        deferred_optional_count = 0
        candidate_packet_ids = {packet_id for packet_id, _edge, _packet_key in candidates}

        for packet_id, edge, _packet_key in candidates:
            from_uri = resolve_uri(edge.get("from_uri")) or resolve_identity(
                edge.get("from_identity")
            )
            relation = edge.get("relation", edge.get("field"))
            relation_name = relation.strip() if isinstance(relation, str) else ""
            if from_uri is None or not relation_name:
                unresolved_count += 1
                continue
            from_info = known_info_by_uri.get(from_uri)
            if from_info is None or relation_name not in set(
                self._relation_field_names(from_info[0])
            ):
                unresolved_count += 1
                continue
            if scope and from_uri not in scope:
                continue
            if is_optional_relation_field(from_info[0], relation_name):
                deferred_optional_count += 1
                continue
            target_uri = resolve_uri(edge.get("to_uri")) or resolve_identity(
                edge.get("to_identity")
            )
            if target_uri is None:
                unresolved_count += 1
                continue

            concept = from_info[0]
            group = groups.setdefault(
                concept,
                {
                    "concept": concept,
                    "entity_uris": set(),
                    "packet_ids": set(),
                    "packet_ids_by_entity": {},
                    "candidate_count_by_entity": {},
                    "relation_fields": set(),
                    "candidate_count": 0,
                },
            )
            entity_set = group["entity_uris"]
            packet_set = group["packet_ids"]
            relation_set = group["relation_fields"]
            assert isinstance(entity_set, set)
            assert isinstance(packet_set, set)
            assert isinstance(relation_set, set)
            packet_ids_by_entity = group["packet_ids_by_entity"]
            candidate_count_by_entity = group["candidate_count_by_entity"]
            assert isinstance(packet_ids_by_entity, dict)
            assert isinstance(candidate_count_by_entity, dict)
            entity_set.add(from_uri)
            packet_set.add(packet_id)
            entity_packet_ids = packet_ids_by_entity.setdefault(from_uri, set())
            entity_candidate_count = candidate_count_by_entity.get(from_uri, 0)
            assert isinstance(entity_packet_ids, set)
            assert isinstance(entity_candidate_count, int)
            entity_packet_ids.add(packet_id)
            candidate_count_by_entity[from_uri] = entity_candidate_count + 1
            relation_set.add(relation_name)
            group["candidate_count"] = int(group["candidate_count"]) + 1

        shards: list[dict[str, object]] = []
        all_entity_uris: set[str] = set()
        shard_index = 0
        for concept in sorted(groups):
            group = groups[concept]
            entity_set = group["entity_uris"]
            packet_set = group["packet_ids"]
            relation_set = group["relation_fields"]
            assert isinstance(entity_set, set)
            assert isinstance(packet_set, set)
            assert isinstance(relation_set, set)
            packet_ids_by_entity = group["packet_ids_by_entity"]
            candidate_count_by_entity = group["candidate_count_by_entity"]
            assert isinstance(packet_ids_by_entity, dict)
            assert isinstance(candidate_count_by_entity, dict)
            entity_list = sorted(str(uri) for uri in entity_set)
            for start in range(0, len(entity_list), _entity_batch_limit()):
                shard_index += 1
                entity_chunk = entity_list[start : start + _entity_batch_limit()]
                packet_list = sorted(
                    {
                        str(packet_id)
                        for entity_uri in entity_chunk
                        for packet_id in packet_ids_by_entity.get(entity_uri, set())
                    },
                )
                candidate_count = sum(
                    int(candidate_count_by_entity.get(entity_uri, 0)) for entity_uri in entity_chunk
                )
                all_entity_uris.update(entity_chunk)
                shards.append(
                    {
                        "shard_id": f"relation_{concept.casefold()}_{shard_index}",
                        "concept": concept,
                        "entity_uris": entity_chunk,
                        "write_set": entity_chunk,
                        "packet_ids": packet_list,
                        "relation_fields": sorted(str(field) for field in relation_set),
                        "candidate_count": candidate_count,
                        "sync_component_links": False,
                    },
                )

        join = {
            "required": bool(shards),
            "entity_uris": sorted(all_entity_uris),
            "write_set": sorted(all_entity_uris),
            "packet_ids": sorted(candidate_packet_ids),
            "sync_component_links": True,
            "depends_on_shards": [str(shard["shard_id"]) for shard in shards],
        }
        if global_scope:
            self.store.write_json(
                self._relation_manifest_key(str(checkpoint["build_id"])),
                {
                    "version": 1,
                    "build_id": str(checkpoint["build_id"]),
                    "doc_id": str(checkpoint.get("doc_id", "")),
                    "input_docs": checkpoint.get("input_docs", []),
                    "packet_ids": sorted(manifest_packet_ids),
                    "candidate_packet_ids": sorted(candidate_packet_ids),
                    "candidate_entity_uris": sorted(all_entity_uris),
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
        return {
            "shards": shards,
            "shard_count": len(shards),
            "candidate_count": len(candidates),
            "manifest_packet_ids": sorted(manifest_packet_ids),
            "candidate_packet_ids": sorted(candidate_packet_ids),
            "global_scope": global_scope,
            "unresolved_count": unresolved_count,
            "deferred_optional_count": deferred_optional_count,
            "join": join,
            "idempotent": True,
        }

    def build_relation_closure(
        self,
        entity_uris: list[str] | None = None,
        *,
        packet_ids: list[str] | None = None,
        sync_component_links: bool = True,
    ) -> dict[str, object]:
        """Resolve source-packet relation candidates after materialization.

        The operation is deterministic and idempotent.  It consumes
        ``local_relation_edges`` (or the compatibility field
        ``cross_concept_handoffs``) from source packets, resolves exact
        natural keys once all entities are visible, merges each relation field
        with optimistic locking, and materializes body links in one closure
        pass.  Unresolved candidates are returned as stable records for OPA or
        a later build; they are never converted into guessed URIs.
        """
        checkpoint = self.inspect_build_checkpoint()
        stage = str(checkpoint.get("stage", ""))
        if not checkpoint.get("exists") or stage not in _RELATION_CLOSURE_READY_STAGES:
            current_stage = stage or "missing"
            raise ValueError(
                f"Relation closure is gated until materialization is complete; current checkpoint stage={current_stage!r}. Call checkpoint_build(stage='materialized') first.",
            )

        relation_started = time.perf_counter()
        scope = {
            uri.split("#", 1)[0]
            for uri in (entity_uris or [])
            if isinstance(uri, str) and uri.strip()
        }
        manifest_packet_ids = self._relation_manifest_packet_ids(checkpoint)
        selected_packets = {
            packet_id.strip() for packet_id in (packet_ids or []) if packet_id.strip()
        }
        if packet_ids is None:
            selected_packets = set(manifest_packet_ids)
        known_info_by_uri, resolve_uri, resolve_identity, candidates = self._relation_context(
            selected_packets
        )

        relation_scope_complete = True
        coverage_missing_packets: set[str] = set()
        coverage_missing_entities: set[str] = set()
        if sync_component_links and manifest_packet_ids:
            _all_info, all_resolve_uri, all_resolve_identity, all_candidates = (
                self._relation_context(manifest_packet_ids)
            )
            required_packet_ids = {packet_id for packet_id, _edge, _packet_key in all_candidates}
            coverage_missing_packets = required_packet_ids - selected_packets
            for _packet_id, edge, _packet_key in all_candidates:
                from_uri = all_resolve_uri(edge.get("from_uri")) or all_resolve_identity(
                    edge.get("from_identity")
                )
                relation = edge.get("relation", edge.get("field"))
                relation_name = relation.strip() if isinstance(relation, str) else ""
                if from_uri is None or not relation_name or from_uri not in _all_info:
                    continue
                if is_optional_relation_field(_all_info[from_uri][0], relation_name):
                    continue
                coverage_missing_entities.add(from_uri)
            if scope:
                coverage_missing_entities.difference_update(scope)
            else:
                coverage_missing_entities.clear()
            relation_scope_complete = not coverage_missing_packets and not coverage_missing_entities
            if not relation_scope_complete:
                missing_packets = sorted(coverage_missing_packets)
                missing_entities = sorted(coverage_missing_entities)
                raise ValueError(
                    "Relation join scope is incomplete for the current build; "
                    f"missing_packets={missing_packets[:10]}, missing_entities={missing_entities[:10]}. "
                    "Run plan_relation_shards() without a scope and complete every returned shard before join.",
                )

        updates: dict[str, dict[str, set[str]]] = {}
        unresolved: list[dict[str, object]] = []
        touched_components: set[str] = set()
        linked_edge_count = 0
        deferred_optional_count = 0
        for packet_id, edge, packet_key in candidates:
            from_uri = resolve_uri(edge.get("from_uri")) or resolve_identity(
                edge.get("from_identity")
            )
            relation = edge.get("relation", edge.get("field"))
            relation_name = relation.strip() if isinstance(relation, str) else ""
            if from_uri is None or not relation_name:
                unresolved.append(
                    {
                        "packet_id": packet_id,
                        "packet_key": packet_key,
                        "from_uri": from_uri or edge.get("from_uri", ""),
                        "relation": relation_name,
                        "to_uri": edge.get("to_uri", ""),
                        "to_identity": edge.get("to_identity", {}),
                        "status": str(edge.get("status", "needs_predecessor")),
                        "reason": "source_or_target_not_materialized_or_relation_missing",
                    },
                )
                continue
            from_info = known_info_by_uri.get(from_uri) or self.store.lookup_by_uri(from_uri)
            if from_info is None or relation_name not in set(
                self._relation_field_names(from_info[0])
            ):
                unresolved.append(
                    {
                        "packet_id": packet_id,
                        "packet_key": packet_key,
                        "from_uri": from_uri,
                        "relation": relation_name,
                        "to_uri": edge.get("to_uri", ""),
                        "status": "invalid_relation_field",
                        "reason": "relation_not_declared_by_source_concept_schema",
                    },
                )
                continue
            if scope and from_uri not in scope:
                continue
            if is_optional_relation_field(from_info[0], relation_name):
                deferred_optional_count += 1
                continue
            target_uri = resolve_uri(edge.get("to_uri")) or resolve_identity(
                edge.get("to_identity")
            )
            if target_uri is None:
                unresolved.append(
                    {
                        "packet_id": packet_id,
                        "packet_key": packet_key,
                        "from_uri": from_uri,
                        "relation": relation_name,
                        "to_uri": edge.get("to_uri", ""),
                        "to_identity": edge.get("to_identity", {}),
                        "status": str(edge.get("status", "needs_predecessor")),
                        "reason": "source_or_target_not_materialized_or_relation_missing",
                    },
                )
                continue
            updates.setdefault(from_uri, {}).setdefault(relation_name, set()).add(target_uri)
            target_info = known_info_by_uri.get(target_uri) or self.store.lookup_by_uri(target_uri)
            if target_info is not None and target_info[0] == "Component":
                touched_components.add(target_uri)
            linked_edge_count += 1

        patches: list[dict[str, object]] = []
        skipped: list[str] = []
        update_items = sorted(updates.items())

        def read_update(
            item: tuple[str, dict[str, set[str]]],
        ) -> tuple[str, dict[str, set[str]], str | None]:
            uri, fields = item
            return uri, fields, self.store.read_entity_by_uri(uri)

        with ThreadPoolExecutor(
            max_workers=min(_io_worker_limit(), max(1, len(update_items)))
        ) as pool:
            update_records = list(pool.map(read_update, update_items))
        for from_uri, fields, content in update_records:
            info = known_info_by_uri.get(from_uri) or self.store.lookup_by_uri(from_uri)
            if info is None or content is None:
                continue
            concept, class_name, object_name = info
            frontmatter = parse_frontmatter(content)
            operations: list[dict[str, object]] = []
            for field, targets in sorted(fields.items()):
                current = frontmatter.get(field)
                if isinstance(current, list):
                    existing = [item for item in current if isinstance(item, str) and item.strip()]
                    merged = list(dict.fromkeys([*existing, *sorted(targets)]))
                    if merged != existing:
                        operations.append({"op": "fm_set_list", "field": field, "values": merged})
                elif isinstance(current, str) and current.strip():
                    if current.strip() not in targets:
                        unresolved.append(
                            {
                                "from_uri": from_uri,
                                "relation": field,
                                "to_uri": sorted(targets),
                                "status": "cardinality_conflict",
                                "reason": "scalar_relation_already_has_a_different_target",
                            },
                        )
                elif self._relation_field_shape(concept, field) == "ref":
                    target = sorted(targets)[0]
                    if len(targets) > 1:
                        unresolved.append(
                            {
                                "from_uri": from_uri,
                                "relation": field,
                                "to_uri": sorted(targets),
                                "status": "cardinality_conflict",
                                "reason": "schema_declares_scalar_relation_but_candidates_are_multiple",
                            },
                        )
                    else:
                        operations.append({"op": "fm_set", "field": field, "value": target})
                else:
                    operations.append({
                        "op": "fm_set_list",
                        "field": field,
                        "values": sorted(targets),
                    })
            if not operations:
                skipped.append(from_uri)
                continue
            current_hash = sha256(content.encode("utf-8")).hexdigest()
            patches.append(
                {
                    "concept": concept,
                    "class_name": class_name or "",
                    "object_name": object_name,
                    "operations": operations,
                    "expected_sha256": current_hash,
                },
            )

        preloaded = {
            uri: content for uri, _fields, content in update_records if content is not None
        }
        patched: list[str] = []
        batch_limit = _entity_batch_limit()
        for start in range(0, len(patches), batch_limit):
            result = self.patch_entities_batch(
                patches[start : start + batch_limit],
                sync_component_links=sync_component_links,
                preloaded_contents=preloaded,
            )
            patched.extend(str(uri) for uri in result.get("uris", []) if isinstance(uri, str))

        if sync_component_links:
            for component_uri in sorted(touched_components):
                self._sync_component_narrative_links(component_uri, force=True)
        device_diagnostic_page_count = (
            self._sync_device_diagnostic_links() if sync_component_links else 0
        )
        self._record_phase_timing("relation", relation_started)
        # Advance checkpoint so restart can distinguish "relations done" from
        # "merely materialized"; relation_closure is idempotent and in the
        # ready-set, so re-calls are safe.
        self.checkpoint_build(
            doc_id=str(checkpoint.get("doc_id", "")),
            device_id=str(checkpoint.get("device_id", "")),
            series_id=str(checkpoint.get("series_id", "")),
            stage="relation_closure",
            build_id=str(checkpoint.get("build_id", "")),
            input_hash=str(checkpoint.get("input_hash", "")),
            config_hash=str(checkpoint.get("config_hash", "")),
            source_snapshot_id=str(checkpoint.get("source_snapshot_id", "")),
            snapshot_id=str(checkpoint.get("snapshot_id", "")),
            input_docs=tuple(checkpoint.get("input_docs", ()) or ()),
            schema_version=str(checkpoint.get("schema_version", "")),
            audit_profile=self._validate_audit_profile(
                str(checkpoint.get("audit_profile", "manual")),
            ),
        )
        return {
            "candidate_count": len(candidates),
            "linked_edge_count": linked_edge_count,
            "patched_entity_count": len(patched),
            "skipped_entity_count": len(skipped),
            "deferred_optional_count": deferred_optional_count,
            "relation_scope_complete": relation_scope_complete,
            "manifest_packet_count": len(manifest_packet_ids),
            "selected_packet_count": len(selected_packets),
            "device_diagnostic_page_count": device_diagnostic_page_count,
            "patched_entity_uris": patched,
            "unresolved": unresolved,
            "idempotent": True,
        }
