"""Search, backlinks, change report, and finalize."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from hashlib import sha256
import json
import logging
import os
import re
import time

from wolfharness.capabilities.wiki.io.text_parsers import (
    _parse_forward_links,
)
from wolfharness.capabilities.wiki.quality import (
    BuildProfile,
    WikiAuditReport,
    entity_status,
    extract_source_uris,
    parse_frontmatter,
)
from wolfharness.capabilities.wiki.schema_loader import get_schema_version


logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING

from wolfharness.capabilities.wiki._helpers import _with_publication_state


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


# Core-path publication blockers: the diagnostic backbone a Wiki page
# dead-ends without.  Mechanism content (Component.working_mechanism,
# Fault.failure_mechanism) and the required Device↔Symptom↔Fault graph edges
# (Device.critical_components, SymptomProfile.device_refs, DTC.related_faults)
# hard-block finalize.  Every other error — Procedure associations,
# body-link completeness, isolated/draft entities, hook.* structural findings
# — is reported but never stops publication.  This mirrors the OPA/planning
# backbone policy (``is_optional_relation_issue``) that already defers
# non-required relations; the finalize gate must not self-lock on relation
# edges the schema itself treats as optional.
_CORE_PATH_BLOCKER_CODES = frozenset({
    "Component.working_mechanism",  # content: body ## 工作机理
    "Fault.failure_mechanism",  # content: body ## 失效机理
    "Device.critical_components",  # relation: frontmatter (BOM-filled) + body fallback
    "DTC.related_faults",  # relation: frontmatter OR body ## 可能失效机理
    # ponytail: SymptomProfile.device_refs removed — schema-optional, no body
    # fallback, closure defers it.  Blocking on it contradicts the backbone
    # policy and body-first design.  DTC→Device has no field-level code
    # (bound via class_name/controller), so DTC.related_faults is the only
    # DTC link code the audit can raise.
})


class FinalizeMixin:
    """Search, backlinks, change report, and finalize."""

    def grep_wiki(self, pattern: str, *, limit: int = 256, target_uri: str = "") -> list[dict]:
        """Regex text search across all wiki entities.

        Returns matching entities with ``key``, ``uri``, ``score``, ``abstract``.
        On local backends returns an empty list.
        """
        if target_uri:
            allowed = (self.store.root_uri, self._raw_fs.root_uri)
            target = target_uri.rstrip("/")
            if not any(target == root or target.startswith(root + "/") for root in allowed):
                raise ValueError(
                    "target_uri must belong to the configured wiki or raw OpenViking root"
                )
        return self.store._fs.grep(pattern, limit=limit, target_uri=target_uri)

    # ── Backlinks ───────────────────────────────────────────────────────────

    def rebuild_backlinks(self, entity_uris_with_links: list[tuple[str, list[str]]]) -> int:
        """Rebuild ``backlinks_index.json`` from a complete set of entity→links pairs.

        self._invalidate_audit_cache()
        Returns the number of backlink entries written.
        """
        if not entity_uris_with_links:
            return 0
        self.store.rebuild_backlinks(entity_uris_with_links)
        return len(entity_uris_with_links)

    def rebuild_all_backlinks(self) -> dict[str, object]:
        """Reconcile deterministic graph projections and rebuild backlinks.

        Relation workers only patch their disjoint source entities. Device
        diagnostic navigation is a deterministic projection of those links,
        so it is synchronized here in-process instead of scheduling a fragile
        global join worker.
        """
        device_diagnostic_page_count = self._sync_device_diagnostic_links()
        component_narrative_page_count = self._sync_all_component_narrative_links()
        pairs: list[tuple[str, list[str]]] = []
        for concept, _class_name, _object_name, uri in self.store.list_entities():
            content = self.store.read_entity_by_uri(uri)
            if content:
                linked = [u for u in _parse_forward_links(content) if u != uri]
                pairs.append((uri, linked))
            if concept == "Symptom":
                for profile in self.list_symptom_profiles(uri):
                    profile_uri = profile["uri"]
                    profile_content = self.read_resource(profile_uri)
                    if profile_content:
                        linked = [
                            lu for lu in _parse_forward_links(profile_content) if lu != profile_uri
                        ]
                        pairs.append((profile_uri, linked))
        count = self.rebuild_backlinks(pairs)
        return {
            "backlink_entries": count,
            "device_diagnostic_page_count": device_diagnostic_page_count,
            "component_narrative_page_count": component_narrative_page_count,
            "native_relation_sync": self.store.native_relation_sync_result
            or {"status": "not_attempted"},
        }

    def _purge_build_intermediates(self) -> dict[str, object]:
        """Delete build-only intermediate state after a successful finalize.

        ``chapter_plans``, ``source_packets``, ``relation_work``, and
        ``relation_manifests`` are consumed during the pipeline and no longer
        needed once the wiki is finalized.  ``materialization_receipts``
        (audit trail) and ``build_checkpoint.json`` (completion record) are
        preserved.
        """
        purged: dict[str, int] = {}
        for dir_key in (
            "source_packets",
            "index/chapter_plans",
            "index/relation_work",
            "index/relation_manifests",
        ):
            files = self.store.list_dir(dir_key, recursive=True)
            count = 0
            for file_key in files:
                if file_key.endswith(".json"):
                    self.store.delete(file_key)
                    count += 1
            self.store.remove_empty_dir(dir_key)
            purged[dir_key] = count
        self._invalidate_audit_cache()
        logger.info("Purged build intermediates after finalize: %s", purged)
        return {"purged_intermediates": purged}

    def get_backlinks(self, uri: str) -> list[str]:
        """Return URIs that link to *uri* (empty if none / no index)."""
        return self.store.get_backlinks(uri)

    def prune_stale_index_entries(self) -> dict[str, object]:
        """Compatibility no-op after entity identity indexes were retired."""
        self._invalidate_audit_cache()
        return {
            "removed_count": 0,
            "removed_uris": [],
            "status": "identity_indexes_retired",
        }

    # ── Wiki structure finalization (resource.json, concepts index, etc.) ────
    #
    # Ported from builder_legacy.py `_write_to_disk_unlocked` — these methods
    # build the wiki directory structure artifacts AFTER all entities have been
    # written.  Every filesystem op goes through WikiStore.

    def entity_uri(self, concept: str, class_name: str | None, object_name: str) -> str:
        """Build the canonical readable ``{root_uri}/<Concept>/<Class>/<Object>`` URI."""
        return self.store.entity_uri(concept, class_name, object_name)

    def write_resource_manifest(
        self,
        doc_id: str,
        device_id: str,
        series_id: str,
        entities: list[dict],
        *,
        authoritative: bool = False,
        build_id: str = "",
        snapshot_id: str = "",
        input_snapshot_hash: str = "",
        config_hash: str = "",
        source_snapshot_id: str = "",
        input_docs: tuple[str, ...] = (),
        spec_version: str = "",
        persist: bool = True,
    ) -> dict:
        """Write ``resource.json`` — source URI → 文件路径映射.

        ``entities`` is a list of dicts with keys:
        ``concept, class_name, object_name, title, description, uri``.
        Merges with existing resource.json (preserving other models' entries).
        """
        spec_version = spec_version or get_schema_version()
        # Dedup by URI + path (same entity written multiple times via merge).
        seen: set[tuple[str, str]] = set()
        resources: list[dict] = []
        for ent in entities:
            uri = ent.get("uri", "")
            if not uri:
                continue
            concept = ent.get("concept", "")
            path_value = ent.get("path")
            if isinstance(path_value, str) and path_value:
                path = path_value
            else:
                path = self.store._key_of(
                    self.store.entity_path(
                        concept,
                        ent.get("class_name") or None,
                        ent.get("object_name", ""),
                    ),
                )
            key = (uri, path)
            if key in seen:
                continue
            seen.add(key)
            # A leaf hash/name is not globally unique across manuals.  Use the
            # complete URI so SY75 and SY215C entries cannot overwrite one
            # another in the merged resource manifest.
            resource_id = sha256(uri.encode("utf-8")).hexdigest()[:24] if uri else ""
            resources.append(
                {
                    "resource_id": resource_id,
                    "kind": f"entity_{concept.lower()}",
                    "uri": uri,
                    "path": path,
                    "mime_type": "text/markdown",
                    "title": ent.get("title", ent.get("object_name", "")),
                    "name": ent.get("object_name", ""),
                    "entity_type": concept,
                },
            )

        manifest = {
            "version": 2,
            "spec_version": spec_version,
            "build_id": build_id,
            "snapshot_id": snapshot_id,
            "input_snapshot_hash": input_snapshot_hash,
            "config_hash": config_hash,
            "input_docs": list(input_docs),
            "model_id": device_id,
            "uri": f"{self.store.root_uri}/resource/{device_id}.json",
            "resources": resources,
        }

        # Merge with existing resource.json (preserve other models' entries).
        existing: dict = {}
        existing_raw = self.store.read_text("resource.json")
        if existing_raw is not None and not authoritative:
            try:
                parsed = json.loads(existing_raw)
                if isinstance(parsed, dict):
                    existing = parsed
            except json.JSONDecodeError:
                pass
        old_resources = existing.get("resources") if isinstance(existing, dict) else None
        if isinstance(old_resources, list):
            # Merge by resource_id — keep ALL existing entries, overwrite
            # same resource_ids from current batch (handles multiple docs
            # with the same device_id, e.g. SY215C维修手册 + fault-cases).
            merged = {
                str(e.get("resource_id")): e
                for e in old_resources
                if isinstance(e, dict) and e.get("resource_id")
            }
            merged.update(
                {str(e["resource_id"]): e for e in resources if e.get("resource_id")},
            )
            manifest["resources"] = list(merged.values())

        if persist:
            self.store.write_text(
                "resource.json",
                json.dumps(manifest, ensure_ascii=False, indent=2),
            )
        logger.info("resource.json: %d resources indexed", len(manifest["resources"]))
        return manifest

    def write_concepts_index(
        self,
        device_id: str,
        entities: list[dict],
        *,
        authoritative: bool = False,
        persist: bool = True,
    ) -> dict:
        """Group entities by concept and write a concepts index.

        Writes ``index/concepts_index.json`` — a ``{concept_name: [entity_uri, ...]}``
        map so downstream consumers can enumerate entities by concept.
        Merges with existing index (handles multi-doc builds).
        """
        # Load existing index.
        groups: dict[str, list[str]] = {}
        existing_raw = self.store.read_text("index/concepts_index.json")
        if existing_raw is not None and not authoritative:
            try:
                parsed = json.loads(existing_raw)
                if isinstance(parsed, dict):
                    groups = parsed
            except json.JSONDecodeError:
                pass

        seen: set[tuple[str, str]] = set()
        for concept, uris in groups.items():
            for u in uris:
                seen.add((concept, u))
        for ent in entities:
            concept = ent.get("concept", "")
            uri = ent.get("uri", "")
            if concept and uri and (concept, uri) not in seen:
                seen.add((concept, uri))
                groups.setdefault(concept, []).append(uri)

        # Write merged concepts_index.json.
        if persist:
            self.store.write_text(
                "index/concepts_index.json",
                json.dumps(groups, ensure_ascii=False, indent=2),
            )

        logger.info("Concepts index: %d concepts, %d entities", len(groups), len(entities))
        return groups

    def build_change_report(
        self,
        *,
        persist: bool = True,
        limit: int = 10000,
        include_op_flow: bool = True,
        audit_report: WikiAuditReport | None = None,
    ) -> dict[str, object]:
        """Summarize the current incremental build and its publication evidence.

        The report is deterministic: entity changes come from the structured
        build JSONL, while current counts and content summaries come from the
        committed Wiki store.  ``event_span_seconds`` is deliberately named
        to avoid presenting the first/last logged event span as a full process
        wall-clock measurement when a caller did not attach a logger.
        """
        if limit < 1 or limit > 10000:
            raise ValueError("build_change_report limit must be between 1 and 10000")
        checkpoint = self.store.read_json("index/build_checkpoint.json")
        build_started_at = (
            str(checkpoint.get("started_at", "")).strip() if isinstance(checkpoint, dict) else ""
        )
        log_paths: list[Path] = []
        if self._log:
            log_paths = sorted(self._log.log_dir.glob("wiki_build_*.jsonl"))
            if self._log.path not in log_paths:
                log_paths.append(self._log.path)
        events: list[dict[str, object]] = []
        for log_path in log_paths:
            if not log_path.exists():
                continue
            for line in log_path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                if build_started_at:
                    event_time = str(record.get("timestamp", ""))
                    if not event_time or event_time < build_started_at:
                        continue
                events.append(record)
        entity_events = [
            event for event in events if event.get("event") in {"entity_created", "entity_updated"}
        ]
        entity_events = entity_events[-limit:]
        formal_records = self._formal_entity_snapshot_records()
        counts_by_concept: dict[str, int] = {}
        for record in formal_records:
            concept = str(record.get("concept", ""))
            counts_by_concept[concept] = counts_by_concept.get(concept, 0) + 1

        changed_entities: list[dict[str, object]] = []
        for event in entity_events:
            uri = str(event.get("uri", ""))
            content = self.read_resource(uri) if uri else None
            frontmatter = parse_frontmatter(content or "") if content else {}
            changed_entities.append(
                {
                    "event": event.get("event", ""),
                    "timestamp": event.get("timestamp", ""),
                    "uri": uri,
                    "concept": event.get("concept", ""),
                    "class_name": event.get("class_name", ""),
                    "object_name": event.get("object_name", ""),
                    "char_count_before": event.get("char_count_before", 0),
                    "char_count_after": event.get("char_count", len(content or "")),
                    "reason": event.get("reason", ""),
                    "source_uris": list(extract_source_uris(content or "")),
                    "applicable_models": frontmatter.get("applicable_models", []),
                    "conflict_pending": str(frontmatter.get("conflict_pending", "")).lower()
                    in {"true", "yes", "1"},
                    "sections": re.findall(r"^##\s+(.+?)\s*$", content or "", re.MULTILINE),
                },
            )

        event_timestamps: list[datetime] = []
        for event in events:
            timestamp = event.get("timestamp")
            if not isinstance(timestamp, str):
                continue
            try:
                event_timestamps.append(datetime.fromisoformat(timestamp))
            except ValueError:
                continue
        event_span_seconds = (
            round((max(event_timestamps) - min(event_timestamps)).total_seconds(), 3)
            if len(event_timestamps) >= 2
            else None
        )
        build_elapsed_seconds: float | None = None
        if isinstance(checkpoint, dict):
            checkpoint_times: list[datetime] = []
            for key in ("started_at", "updated_at"):
                timestamp = checkpoint.get(key)
                if not isinstance(timestamp, str):
                    continue
                try:
                    checkpoint_times.append(datetime.fromisoformat(timestamp))
                except ValueError:
                    continue
            if len(checkpoint_times) == 2:
                build_elapsed_seconds = round(
                    (checkpoint_times[1] - checkpoint_times[0]).total_seconds(),
                    3,
                )
        created_count = sum(1 for event in entity_events if event.get("event") == "entity_created")
        updated_count = sum(1 for event in entity_events if event.get("event") == "entity_updated")
        mutation_attempt_count = sum(
            1 for event in events if event.get("event") == "mutation_attempt"
        )
        mutation_applied_count = sum(
            1 for event in events if event.get("event") == "mutation_applied"
        )
        source_packet_unique_count = len(
            {
                str(event.get("packet_id", ""))
                for event in events
                if event.get("event") == "source_packet_recorded"
                and str(event.get("packet_id", ""))
            },
        )
        by_change_concept: dict[str, dict[str, int]] = {}
        for event in entity_events:
            concept = str(event.get("concept", ""))
            bucket = by_change_concept.setdefault(concept, {"created": 0, "updated": 0})
            event_name = "created" if event.get("event") == "entity_created" else "updated"
            bucket[event_name] += 1
        phase_wall_ms: dict[str, float | None] = {
            "extraction": None,
            "materialization": None,
            "relation": None,
            "audit": None,
            "finalize": None,
        }
        phase_totals: dict[str, float] = {}
        for event in events:
            if event.get("event") != "phase_timing":
                continue
            phase = str(event.get("phase", "")).strip()
            duration = event.get("duration_ms")
            if phase and isinstance(duration, (int, float)):
                phase_totals[phase] = phase_totals.get(phase, 0.0) + float(duration)
        for phase, duration in phase_totals.items():
            if phase in phase_wall_ms:
                phase_wall_ms[phase] = round(duration, 3)
        issue_counts: dict[str, int] = {}
        structural_valid_count: int | None = None
        content_confirmed_count = sum(
            1
            for record in formal_records
            if entity_status(str(record.get("content", ""))) == "confirmed"
        )
        if audit_report is not None:
            structural_valid_uris = {str(record.get("uri", "")) for record in formal_records}
            audit_issues = audit_report.get("issues", [])
            if not isinstance(audit_issues, list):
                audit_issues = []
            for issue in audit_issues:
                if not isinstance(issue, dict):
                    continue
                if issue.get("severity") == "error":
                    code = str(issue.get("code", ""))
                    issue_counts[code] = issue_counts.get(code, 0) + 1
                    structural_valid_uris.discard(str(issue.get("uri", "")))
            structural_valid_count = len(structural_valid_uris)
        op_flow: dict[str, object]
        if include_op_flow:
            report_build_id = (
                str(checkpoint.get("build_id", "")).strip() if isinstance(checkpoint, dict) else ""
            )
            op_flow = self.op_flow_report(
                persist=persist,
                limit=limit,
                build_id=report_build_id or None,
            )
        else:
            op_flow = {
                "counts": {
                    "opa": len(self.get_opas(limit=limit)),
                    "ops": len(self.get_ops(limit=limit)),
                    "opl": len(self.get_opls(limit=limit)),
                },
                "coverage_skipped": True,
            }
        report: dict[str, object] = {
            "version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "event_log_path": str(self._log.path) if self._log else "",
            "event_count": len(events),
            "event_span_seconds": event_span_seconds,
            "build_elapsed_seconds": build_elapsed_seconds,
            "entity_created_count": created_count,
            "entity_updated_count": updated_count,
            "entity_touched_count": created_count + updated_count,
            "mutation_attempt_count": mutation_attempt_count,
            "mutation_applied_count": mutation_applied_count,
            "source_packet_unique_count": source_packet_unique_count,
            "structural_valid_count": structural_valid_count,
            "content_confirmed_count": content_confirmed_count,
            "publication_blocker_counts": issue_counts,
            "entity_count_current": len(formal_records),
            "entity_count_by_concept": counts_by_concept,
            "changes_by_concept": by_change_concept,
            "changed_entities": changed_entities,
            "phase_wall_ms": phase_wall_ms,
            "op_flow": op_flow,
            "model_mapping": self.model_mapping_report(),
        }
        if persist:
            self.store.write_json("index/build_delta_report.json", report)
        return report

    def finalize_wiki(
        self,
        doc_id: str,
        device_id: str,
        series_id: str,
        entities: list[dict] | None = None,
        *,
        authoritative: bool = False,
        allow_residual_errors: bool = False,
        audit_profile: BuildProfile = "manual",
    ) -> dict:
        """Build wiki structure artifacts after entity extraction.

        Call this once per doc, after all entities are written/merged.
        Returns ``{resource_count, concept_count, entity_count,
        unresolved_opa_count}``.

        ``allow_residual_errors`` is retained for caller compatibility but
        never bypasses blockers. A source-honest relation gap may remain in
        the page/ledger as ``open_gap``; only substantive content/identity
        gaps and fact conflicts become OPA review records. Final pages are
        stamped as machine-validated with ``review_state=unreviewed``.
        """
        finalize_started = time.perf_counter()
        audit_profile = self._validate_audit_profile(audit_profile)
        current_checkpoint = self.store.read_json("index/build_checkpoint.json")
        if current_checkpoint is not None:
            checkpoint_identity = tuple(
                str(current_checkpoint.get(field, ""))
                for field in ("doc_id", "device_id", "series_id")
            )
            requested_identity = (doc_id, device_id, series_id)
            checkpoint_profile = str(current_checkpoint.get("audit_profile", "manual"))
            if checkpoint_identity == requested_identity and checkpoint_profile != audit_profile:
                raise ValueError(
                    f"Build audit profile cannot change between checkpoint and finalize: existing={checkpoint_profile}, requested={audit_profile}",
                )
        # Idempotency: if the checkpoint already records a successful finalize
        # for the same entity snapshot, return immediately instead of re-auditing.
        # Re-auditing after promotion would activate strict hooks on confirmed
        # entities and self-lock the gate (the prior finalize's own stamping
        # turns warnings into errors on the next pass).
        if current_checkpoint is not None:
            ckpt_stage = str(current_checkpoint.get("stage", "")).strip()
            ckpt_snapshot = str(current_checkpoint.get("snapshot_id", "")).strip()
            if ckpt_stage == "finalized" and ckpt_snapshot == self._current_entity_snapshot_id():
                logger.info(
                    "finalize_wiki: build already finalized (build_id=%s, snapshot=%s); returning cached result.",
                    current_checkpoint.get("build_id", ""),
                    ckpt_snapshot,
                )
                return {
                    "status": "already_finalized",
                    "build_id": str(current_checkpoint.get("build_id", "")),
                    "snapshot_id": ckpt_snapshot,
                    "source_snapshot_id": str(current_checkpoint.get("source_snapshot_id", "")),
                    "audit_profile": str(current_checkpoint.get("audit_profile", audit_profile)),
                    "entity_count": int(current_checkpoint.get("entity_count", 0)),
                    "published_count": int(current_checkpoint.get("published_count", 0)),
                    "remote_sync": {"status": "not_required"},
                }
            # ponytail: remote_sync_pending means entities are already promoted
            # and committed locally; only the remote upload failed.  Skip audit
            # and promotion (re-auditing confirmed entities would self-lock the
            # gate via strict hooks) and retry just the upload.
            if (
                ckpt_stage == "remote_sync_pending"
                and ckpt_snapshot == self._current_entity_snapshot_id()
            ):
                logger.info(
                    "finalize_wiki: retrying remote sync only (build_id=%s, snapshot=%s).",
                    current_checkpoint.get("build_id", ""),
                    ckpt_snapshot,
                )
                return self._retry_remote_sync(current_checkpoint, audit_profile)
        device_chapter_sync = self.sync_device_system_chapters(doc_id, device_id)
        self._invalidate_audit_cache()
        audit_started = time.perf_counter()
        audit = self._audit_all_pages(profile=audit_profile, limit=500)
        audit_elapsed_ms = round((time.perf_counter() - audit_started) * 1000, 3)
        if not audit["passed"]:
            error_issues = [i for i in audit.get("issues", []) if i.get("severity") == "error"]
            # Only the core diagnostic backbone hard-blocks finalize: missing
            # mechanism content and the Device↔Symptom↔Fault graph edges the
            # user relies on.  Every other error (Procedure associations,
            # body-link completeness, isolated/draft entities, hook.*
            # structural findings) stays visible in the report but never
            # blocks publication.
            blockers = [
                issue
                for issue in error_issues
                if str(issue.get("code", "")) in _CORE_PATH_BLOCKER_CODES
            ]
            if blockers:
                codes = ", ".join(sorted({str(issue.get("code", "")) for issue in blockers}))
                raise ValueError(
                    f"Wiki quality gate has non-waivable core-path blockers: {codes}. "
                    "Resolve the missing mechanism content or "
                    "Device↔Symptom↔Fault edge before finalizing.",
                )
            if error_issues:
                logger.warning(
                    "Finalize: audit has %d non-core-path errors; reporting and proceeding.",
                    len(error_issues),
                )
        audit_snapshot_id = str(audit.get("snapshot_id", ""))
        source_snapshot_id = str(audit.get("source_snapshot_id", ""))
        if not source_snapshot_id:
            raise ValueError("Wiki audit did not produce a raw-source snapshot id.")
        if audit_snapshot_id and audit_snapshot_id != self._current_entity_snapshot_id():
            raise ValueError(
                "Wiki changed after audit; rerun audit_wiki before finalize so publication uses the same entity snapshot.",
            )
        if source_snapshot_id != self._current_source_snapshot_id():
            raise ValueError(
                "Raw sources changed after audit; rerun audit_wiki before finalize so publication uses the same source snapshot.",
            )
        indexed_entities = [
            {
                "concept": record["concept"],
                "class_name": record["class_name"],
                "object_name": record["object_name"],
                "title": record["object_name"],
                "description": "",
                "uri": record["uri"],
                "path": record["path"],
            }
            for record in self._formal_entity_snapshot_records()
        ]
        provided_uris = {
            str(entity.get("uri", "")) for entity in (entities or []) if entity.get("uri")
        }
        indexed_uris = {str(entity["uri"]) for entity in indexed_entities}
        if provided_uris and provided_uris != indexed_uris:
            missing = sorted(indexed_uris - provided_uris)
            extra = sorted(provided_uris - indexed_uris)
            raise ValueError(
                f"Finalize entity set does not match the audited index: "
                f"indexed={len(indexed_uris)}, provided={len(provided_uris)}, "
                f"missing({len(missing)})={missing}, extra({len(extra)})={extra}.",
            )
        # The store/index is the source of truth. Never publish an arbitrary
        # caller-supplied subset after auditing a different entity set.
        entities = indexed_entities
        current_build_id = (
            str(current_checkpoint.get("build_id", "")).strip()
            if isinstance(current_checkpoint, dict)
            else ""
        )
        unresolved = self._unresolved_opa_records(build_id=current_build_id or None)
        # OPA/OPS are review sidecars, not publication gates. The authoritative
        # blocking checks are the audited entity/index checks above. Keep the
        # scoped snapshot for the finalize receipt and observability, but do not
        # reject a build because a review item is pending or legacy data has an
        # unsupported category.
        # OPS is an unconfirmed business-review artifact. Keep the snapshot
        # in the finalize receipt for observability, but never require it
        # before publishing the Wiki.
        op_flow = self.op_flow_status()
        input_doc_ids = self._library_doc_ids() or (doc_id,)
        input_snapshot_hash = self.input_snapshot_hash(input_doc_ids)
        config_hash = self._materialization_config_hash()
        code_revision = self._code_revision()
        completed_checkpoint = self._build_checkpoint_record(
            doc_id,
            device_id,
            series_id,
            "finalized",
            input_hash=input_snapshot_hash,
            config_hash=config_hash,
            snapshot_id=audit_snapshot_id,
            source_snapshot_id=source_snapshot_id,
            input_docs=input_doc_ids,
            schema_version=get_schema_version(),
            audit_profile=audit_profile,
        )
        build_id = str(completed_checkpoint["build_id"])
        started_at = str(completed_checkpoint.get("started_at", datetime.now(UTC).isoformat()))
        manifest = self.write_resource_manifest(
            doc_id,
            device_id,
            series_id,
            entities,
            authoritative=authoritative,
            build_id=build_id,
            snapshot_id=audit_snapshot_id,
            input_snapshot_hash=input_snapshot_hash,
            config_hash=config_hash,
            source_snapshot_id=source_snapshot_id,
            input_docs=input_doc_ids,
            spec_version=get_schema_version(),
            persist=False,
        )
        concepts = self.write_concepts_index(
            device_id,
            entities,
            authoritative=authoritative,
            persist=False,
        )
        current_records = self._formal_entity_snapshot_records()
        page_writes = self._publication_page_writes(current_records)
        publication_allowed_count = sum(
            1
            for record in current_records
            if str(parse_frontmatter(str(record.get("content", ""))).get("status", "")).strip()
            != "deprecated"
        )
        # ``published_count`` is a state count; ``publication_update_count``
        # is the number of files whose publication stamp changed in this run.
        # Keeping both prevents the old metric from confusing "changed now"
        # with "published after finalize".
        published_count = publication_allowed_count
        delta_report = self.build_change_report(
            persist=False,
            include_op_flow=False,
            audit_report=audit,
        )
        op_flow_summary = delta_report.get("op_flow", {})
        op_flow_counts = (
            op_flow_summary.get("counts", {}) if isinstance(op_flow_summary, dict) else {}
        )
        mapping_summary = delta_report.get("model_mapping", {})
        mapping_counts = (
            {
                key: mapping_summary.get(key, 0)
                for key in (
                    "mapping_count",
                    "confirmed_count",
                    "pending_count",
                    "source_backed_count",
                    "unresolved_device_count",
                )
            }
            if isinstance(mapping_summary, dict)
            else {}
        )
        measured_phases = delta_report.get("phase_wall_ms")
        phase_wall_ms: dict[str, float | None] = {
            "extraction": None,
            "materialization": None,
            "relation": None,
            "audit": audit_elapsed_ms,
            "finalize": round((time.perf_counter() - finalize_started) * 1000, 3),
        }
        if isinstance(measured_phases, dict):
            for phase in ("materialization", "relation"):
                value = measured_phases.get(phase)
                if isinstance(value, (int, float)):
                    phase_wall_ms[phase] = float(value)
        completed_at = datetime.now(UTC).isoformat()
        build_elapsed_seconds = round(
            (
                datetime.fromisoformat(completed_at) - datetime.fromisoformat(started_at)
            ).total_seconds(),
            3,
        )
        metrics = {
            "version": 1,
            "status": "finalized",
            "build_id": build_id,
            "doc_id": doc_id,
            "input_docs": list(input_doc_ids),
            "device_id": device_id,
            "series_id": series_id,
            "snapshot_id": audit_snapshot_id,
            "input_snapshot_hash": input_snapshot_hash,
            "config_hash": config_hash,
            "code_revision": code_revision,
            "started_at": started_at,
            "completed_at": completed_at,
            "entity_count": len(entities),
            "resource_count": len(manifest.get("resources", [])),
            "concept_count": len(concepts),
            "published_count": published_count,
            "publication_allowed_count": publication_allowed_count,
            "publication_update_count": len(page_writes),
            "entity_created_count": delta_report.get("entity_created_count", 0),
            "entity_updated_count": delta_report.get("entity_updated_count", 0),
            "entity_touched_count": delta_report.get("entity_touched_count", 0),
            "mutation_attempt_count": delta_report.get("mutation_attempt_count", 0),
            "mutation_applied_count": delta_report.get("mutation_applied_count", 0),
            "source_packet_unique_count": delta_report.get("source_packet_unique_count", 0),
            "structural_valid_count": delta_report.get("structural_valid_count"),
            "content_confirmed_count": int(
                audit.get("confirmed_count", delta_report.get("content_confirmed_count", 0))
            ),
            "publication_blocker_counts": delta_report.get("publication_blocker_counts", {}),
            "event_span_seconds": delta_report.get("event_span_seconds"),
            "build_elapsed_seconds": build_elapsed_seconds,
            "unresolved_opa_count": len(unresolved),
            "audit_error_count": int(audit.get("error_count", 0)),
            "audit_warning_count": int(audit.get("warning_count", 0)),
            "audit_profile": audit_profile,
            "source_snapshot_id": str(audit.get("source_snapshot_id", "")),
            "op_flow_counts": op_flow_counts,
            "model_mapping_counts": mapping_counts,
            "scheduler_mode": os.environ.get("WIKI_SCHEDULER_MODE", "unknown"),
            "gate_mode": "hard_fail",
            "phase_wall_ms": phase_wall_ms,
            "control_plane": {
                "task_count": None,
                "task_retry_count": None,
                "task_reassign_count": None,
                "empty_wakeup_count": None,
                "owner_mismatch_count": None,
                "duplicate_intent_count": None,
                "measurement": "wiki-runtime-only; attach team event sink for task counters",
            },
        }
        control_plane_metrics = metrics.get("control_plane")
        if isinstance(control_plane_metrics, dict):
            metrics.update(
                {
                    key: control_plane_metrics.get(key)
                    for key in (
                        "task_count",
                        "task_retry_count",
                        "task_reassign_count",
                        "empty_wakeup_count",
                        "owner_mismatch_count",
                        "duplicate_intent_count",
                    )
                },
            )
        publication_writes = [
            *page_writes,
            ("resource.json", json.dumps(manifest, ensure_ascii=False, indent=2)),
            ("index/concepts_index.json", json.dumps(concepts, ensure_ascii=False, indent=2)),
            (
                "index/build_metrics.json",
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            ),
            (
                "index/build_checkpoint.json",
                json.dumps(completed_checkpoint, ensure_ascii=False, indent=2) + "\n",
            ),
        ]
        native_relation_sync = self.store.native_relation_sync_result
        if isinstance(native_relation_sync, dict) and native_relation_sync.get("errors"):
            raise ValueError(
                f"OpenViking native relation sync failed; refusing to finalize: {native_relation_sync['errors']}"
            )
        self.store.commit_many(publication_writes)
        self.build_change_report(persist=True)
        # Batch-upload to Viking if using local cache backend. Local commit is
        # recoverable, but remote publication is part of successful finalize:
        # a failed upload returns ``finalized_local`` and leaves a durable
        # ``remote_sync_pending`` checkpoint instead of authorizing phase=done.
        upload_result: dict | None = None
        from wolfharness.capabilities.wiki.storage.local_viking_fs import LocalVikingFS

        if isinstance(self.store._fs, LocalVikingFS):
            from wolfharness.capabilities.wiki.storage import _viking_client

            try:
                executor = ThreadPoolExecutor(max_workers=1)
                try:
                    upload_result = executor.submit(
                        self.store._fs.finalize_upload,
                        _viking_client(),
                    ).result(timeout=200)
                finally:
                    executor.shutdown(wait=False)
                logger.info("Wiki batch-uploaded to Viking: %s", upload_result)
            except Exception:
                logger.exception("Wiki batch-upload to Viking failed; remote sync remains pending")
                upload_result = {"status": "upload_failed"}
        remote_sync_failed = (
            upload_result is not None and upload_result.get("status") != "completed"
        )
        if remote_sync_failed:
            self.store.write_json(
                "index/build_checkpoint.json",
                {
                    **completed_checkpoint,
                    "stage": "remote_sync_pending",
                    "status": "remote_sync_pending",
                    "remote_sync": upload_result,
                },
            )
        self._invalidate_audit_cache()
        # ponytail: purge build-only intermediates after successful finalize so
        # the next ingestion starts clean.  Only purge when remote sync
        # succeeded — on finalized_local (upload failed) keep them for retry.
        purge_result: dict[str, object] = {}
        if not remote_sync_failed:
            purge_result = self._purge_build_intermediates()
        return {
            "resource_count": len(manifest.get("resources", [])),
            "concept_count": len(concepts),
            "entity_count": len(entities),
            "unresolved_opa_count": len(unresolved),
            "published_count": published_count,
            "publication_allowed_count": publication_allowed_count,
            "publication_update_count": len(page_writes),
            "entity_created_count": delta_report.get("entity_created_count", 0),
            "entity_updated_count": delta_report.get("entity_updated_count", 0),
            "entity_touched_count": delta_report.get("entity_touched_count", 0),
            "build_elapsed_seconds": build_elapsed_seconds,
            "build_id": completed_checkpoint["build_id"],
            "snapshot_id": audit_snapshot_id,
            "source_snapshot_id": source_snapshot_id,
            "audit_profile": audit_profile,
            "op_flow_passed": bool(op_flow["passed"]),
            "status": "finalized_local" if remote_sync_failed else "finalized",
            "remote_sync": upload_result or {"status": "not_required"},
            "device_chapter_sync": device_chapter_sync,
            "input_snapshot_hash": input_snapshot_hash,
            "config_hash": config_hash,
            "metrics_path": "index/build_metrics.json",
            "spec_version": get_schema_version(),
            **purge_result,
        }

    def _retry_remote_sync(
        self,
        checkpoint: dict[str, object],
        audit_profile: str,
    ) -> dict[str, object]:
        """Retry only the remote upload when a prior finalize succeeded locally.

        Entities are already promoted and committed locally; re-auditing would
        activate strict hooks on confirmed entities and self-lock the gate.
        """
        from wolfharness.capabilities.wiki.storage.local_viking_fs import LocalVikingFS

        upload_result: dict | None = None
        if isinstance(self.store._fs, LocalVikingFS):
            from wolfharness.capabilities.wiki.storage import _viking_client

            try:
                from concurrent.futures import ThreadPoolExecutor

                executor = ThreadPoolExecutor(max_workers=1)
                try:
                    upload_result = executor.submit(
                        self.store._fs.finalize_upload,
                        _viking_client(),
                    ).result(timeout=3600)
                finally:
                    executor.shutdown(wait=False)
                logger.info("Wiki remote sync retry succeeded: %s", upload_result)
            except Exception:
                logger.exception("Wiki remote sync retry failed")
                upload_result = {"status": "upload_failed"}

        remote_sync_ok = upload_result is None or upload_result.get("status") == "completed"
        if remote_sync_ok:
            # Upgrade checkpoint to finalized
            finalized_checkpoint = {**checkpoint, "stage": "finalized", "status": "finalized"}
            if upload_result:
                finalized_checkpoint["remote_sync"] = upload_result
            self.store.write_json("index/build_checkpoint.json", finalized_checkpoint)
        else:
            self.store.write_json(
                "index/build_checkpoint.json",
                {**checkpoint, "remote_sync": upload_result},
            )

        return {
            "resource_count": 0,
            "concept_count": 0,
            "entity_count": int(checkpoint.get("entity_count", 0)),
            "unresolved_opa_count": 0,
            "published_count": int(checkpoint.get("published_count", 0)),
            "publication_allowed_count": int(checkpoint.get("published_count", 0)),
            "publication_update_count": 0,
            "entity_created_count": 0,
            "entity_updated_count": 0,
            "entity_touched_count": 0,
            "build_id": str(checkpoint.get("build_id", "")),
            "snapshot_id": str(checkpoint.get("snapshot_id", "")),
            "source_snapshot_id": str(checkpoint.get("source_snapshot_id", "")),
            "audit_profile": audit_profile,
            "op_flow_passed": True,
            "status": "finalized" if remote_sync_ok else "finalized_local",
            "remote_sync": upload_result or {"status": "not_required"},
        }

    @staticmethod
    def _publication_page_writes(
        records: Sequence[Mapping[str, object]],
    ) -> list[tuple[str, str]]:
        """Return only pages whose publication fields actually need changing."""
        writes: list[tuple[str, str]] = []
        for record in records:
            path = str(record.get("path", "")).lstrip("/")
            content = str(record.get("content", ""))
            if not path or path.split("/")[0] in {".", ".."}:
                raise ValueError(f"Finalize target escapes wiki root: {path}")
            frontmatter = parse_frontmatter(content)
            deprecated = str(frontmatter.get("status", "")).strip() == "deprecated"
            updated = _with_publication_state(content, deprecated=deprecated)
            if updated != content:
                writes.append((path, updated))
        return writes

    def finalize_wiki_with_migration(
        self,
        doc_id: str,
        device_id: str,
        series_id: str,
        *,
        allow_residual_errors: bool = False,
        audit_profile: BuildProfile = "manual",
    ) -> dict:
        """Migrate legacy profiles, build entity list, then finalize wiki.

        Combines ``migrate_legacy_symptom_profiles`` + entity enumeration +
        ``finalize_wiki`` into a single call so the capability layer has no
        orchestration logic.
        """
        profile_migration = self.migrate_legacy_symptom_profiles()
        all_entities = [
            {
                "concept": record["concept"],
                "class_name": record["class_name"],
                "object_name": record["object_name"],
                "title": record["object_name"],
                "description": "",
                "uri": record["uri"],
                "path": record["path"],
            }
            for record in self._formal_entity_snapshot_records()
        ]
        result = self.finalize_wiki(
            doc_id,
            device_id,
            series_id,
            all_entities,
            authoritative=True,
            allow_residual_errors=allow_residual_errors,
            audit_profile=audit_profile,
        )
        result["migrated_profiles"] = profile_migration["migrated"]
        result["skipped_profiles"] = profile_migration["skipped"]
        return result
