"""Packet, ledger, and source-registration tools."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wolfharness.capabilities.wiki.quality import (
    RawSourceKind,
    classify_raw_source_uri,
)


if TYPE_CHECKING:
    from collections.abc import Mapping


logger = logging.getLogger(__name__)


class PacketBody(BaseModel):
    """Validated packet body for ``record_source_packet``.

    Enforces ``working_mechanism`` when Component candidates are present and
    ``failure_mechanism`` when Fault candidates are present.  BOM identity
    and no-entity packets pass through unchecked via the ``kind`` field.
    """

    model_config = ConfigDict(extra="allow")

    # --- discriminator for internal packet types ---
    kind: str | None = None

    # --- chapter extraction fields (all optional individually) ---
    packet_kind: str | None = None
    source_subject: str = ""
    explicit_facts: list[str] = Field(default_factory=list)
    normalized_identity_candidates: list[dict[str, object]] = Field(default_factory=list)
    entity_candidates: list[dict[str, object]] = Field(default_factory=list)
    entities: list[dict[str, object]] = Field(default_factory=list)
    working_mechanism: str = ""
    failure_mechanism: str = ""
    images: list[dict[str, object]] = Field(default_factory=list)
    dtc: str = ""
    ordered_actions: list[dict[str, object]] = Field(default_factory=list)
    parts_and_specs: object = None
    evidence_map: list[dict[str, object]] = Field(default_factory=list)
    uncertainties: object = None
    cross_concept_handoffs: object = None

    @model_validator(mode="after")
    def _validate_mechanism_fields(self) -> PacketBody:
        """Require mechanism fields when matching candidates exist."""
        # Skip validation for internal packet types
        if self.kind in {"bom_identity_plan", "no_entity"}:
            return self

        # Collect all candidates from the three possible field names
        all_candidates = (
            list(self.normalized_identity_candidates)
            + list(self.entity_candidates)
            + list(self.entities)
        )

        has_component = any(
            str(c.get("concept", "")).lower() == "component" for c in all_candidates
        )
        has_fault = any(str(c.get("concept", "")).lower() == "fault" for c in all_candidates)

        if has_component and not self.working_mechanism.strip():
            raise ValueError(
                "working_mechanism must not be empty when packet contains "
                "Component candidates; infer from diagnostic context and "
                "mark with 'inferred' if the source does not state it explicitly",
            )
        if has_fault and not self.failure_mechanism.strip():
            raise ValueError(
                "failure_mechanism must not be empty when packet contains "
                "Fault candidates; infer from diagnostic context and "
                "mark with 'inferred' if the source does not state it explicitly",
            )

        return self


class PacketMixin:
    """Packet, ledger, and source-registration tools."""

    def source_snapshot(
        self,
        doc_ids: tuple[str, ...] = (),
        *,
        refresh: bool = False,
    ) -> dict[str, str]:
        """Return the current ``raw URI → chapter content hash`` snapshot.

        Raw input is immutable during one build session.  Reuse the first
        snapshot by default; callers explicitly checking for source changes
        pass ``refresh=True`` so the expensive remote walk is not hidden in
        every packet write.
        """
        selected = tuple(sorted(set(doc_ids) or set(self._library_doc_ids(refresh=refresh))))
        if not refresh:
            cached = self._source_snapshot_cache.get(selected)
            if cached is not None:
                return cached
        else:
            self._raw_source_hash_cache.clear()
        source_files: dict[str, str] = {}
        for current_doc_id in selected:
            chapters = self.list_chapters(current_doc_id, refresh=refresh)
            if not chapters:
                # Single-file case document (no chapters/ tree): the source is
                # the case file itself under {doc_id}/, not a chapter leaf.
                doc_prefix = f"{current_doc_id}/"
                for key in self._raw_fs.list_dir(doc_prefix, recursive=True):
                    if not key.endswith(".md"):
                        continue
                    source_files[f"{self._raw_fs.root_uri}/{key}"] = key
                continue
            for chapter in chapters:
                key = str(chapter["md_path"])
                source_uri = f"{self._raw_fs.root_uri}/{key}"
                source_files[source_uri] = key

        fingerprints = {uri: self._raw_fs.fingerprint(key) for uri, key in source_files.items()}
        cached = self._source_snapshot_cache.get(selected)
        cached_fingerprints = self._source_snapshot_fingerprints.get(selected)
        if cached is not None and cached_fingerprints == fingerprints:
            return cached

        snapshots: dict[str, str] = {}
        for source_uri, key in source_files.items():
            content = self._raw_fs.read_text(key)
            if content is None:
                continue
            snapshots[source_uri] = sha256(content.encode("utf-8")).hexdigest()
        result = dict(sorted(snapshots.items()))
        self._source_snapshot_cache[selected] = result
        self._source_snapshot_fingerprints[selected] = fingerprints
        return result

    def _input_snapshot_hash(self, doc_id: str) -> str:
        """Hash the exact source chapters used by one document build."""
        return self.input_snapshot_hash((doc_id,))

    @staticmethod
    def _materialization_config_hash() -> str:
        """Hash schema and taxonomy rules that define materialization behavior."""
        package_root = Path(__file__).resolve().parents[4]
        config_paths = (
            Path(__file__).resolve().parent.parent / "templates" / "default_schema.yaml",
            package_root / "config" / "component-taxonomy.yaml",
            package_root / "config" / "prompts" / "team" / "wiki_extraction_worker.j2",
        )
        content = b"".join(path.read_bytes() if path.is_file() else b"" for path in config_paths)
        return sha256(content).hexdigest()

    @staticmethod
    def _code_revision() -> str:
        """Return an externally supplied immutable code revision when available."""
        explicit = os.environ.get("GIT_COMMIT", os.environ.get("WIKI_CODE_REVISION", "")).strip()
        if explicit:
            return explicit
        source_path = Path(__file__)
        content = source_path.read_bytes() if source_path.is_file() else b""
        return f"local-{sha256(content).hexdigest()[:16]}"

    def _write_build_metrics(self, metrics: dict[str, object]) -> None:
        """Persist a compact durable build summary for restart and operations."""
        self.store.write_text(
            "index/build_metrics.json",
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        )

    def _source_ledger_path(self) -> Path:
        """Return the legacy aggregate ledger path (read-only compatibility)."""
        return self.store.root / "index" / "source_ledger.json"

    @staticmethod
    def _source_packet_key(packet_id: str) -> str:
        """Return the stable business path for one source packet."""
        if re.fullmatch(r"[A-Za-z0-9_]+", packet_id) is None:
            raise ValueError("packet_id must contain only ASCII letters, digits, and underscores")
        return f"source_packets/{packet_id}.json"

    @staticmethod
    def _relation_manifest_key(build_id: str) -> str:
        """Return the durable relation-input manifest key for one build."""
        if not build_id.strip():
            raise ValueError("build_id is required for a relation manifest")
        return f"index/relation_manifests/{build_id}.json"

    @staticmethod
    def _relation_work_key(scope_id: str) -> str:
        """Return the durable work ledger key for one relation scope."""
        if not scope_id.strip():
            raise ValueError("scope_id is required for a relation work ledger")
        return f"index/relation_work/{scope_id}.json"

    @staticmethod
    def _doc_id_variants(doc_id: str) -> set[str]:
        """Return equivalent logical document ids from catalog path variants."""
        normalized = doc_id.strip("/")
        variants = {normalized}
        parts = normalized.split("/") if normalized else []
        while len(parts) >= 2 and parts[-1] == parts[-2]:
            parts.pop()
            variants.add("/".join(parts))
        return variants

    def _packet_matches_build(
        self,
        packet: Mapping[str, object],
        checkpoint: Mapping[str, object],
        allowed_docs: set[str],
    ) -> bool:
        """Match a legacy packet to a build despite nested catalog aliases."""
        packet_doc_id = str(packet.get("doc_id", "")).strip()
        packet_variants = self._doc_id_variants(packet_doc_id)
        if packet_variants & {
            variant for doc_id in allowed_docs for variant in self._doc_id_variants(doc_id)
        }:
            return True
        source_uris = packet.get("source_uris")
        if not isinstance(source_uris, list):
            return False
        checkpoint_docs = {
            str(value).strip()
            for value in [
                checkpoint.get("doc_id"),
                *(
                    checkpoint.get("input_docs", [])
                    if isinstance(checkpoint.get("input_docs"), list)
                    else []
                ),
            ]
            if str(value).strip()
        }
        return any(
            isinstance(source_uri, str)
            and any(self._source_belongs_to_doc(source_uri, doc_id) for doc_id in checkpoint_docs)
            for source_uri in source_uris
        )

    def _chapter_packet_complete_for_build(
        self,
        packet_key: str,
        packet: dict[str, object] | None,
        *,
        build_id: str,
        doc_id: str,
        source_uri: str,
        checkpoint: Mapping[str, object],
    ) -> bool:
        """Return whether a chapter packet is complete for this build.

        Packets written by older workers may omit ``build_id`` even though
        they contain a complete receipt.  The chapter plan and exact source
        URI provide a bounded ownership proof, so adopt that legacy packet
        into the current build instead of rejecting the 1A → 1B transition.
        Packets explicitly owned by another build remain rejected.
        """
        if packet is None or str(packet.get("status", "")) != "complete":
            return False
        source_uris = packet.get("source_uris")
        if not isinstance(source_uris, list) or source_uri not in source_uris:
            return False
        packet_build_id = str(packet.get("build_id", "")).strip()
        if packet_build_id == build_id:
            return True
        if packet_build_id:
            return False
        allowed_docs = {
            doc_id,
            *(
                str(value).strip()
                for value in [
                    checkpoint.get("doc_id"),
                    *(
                        checkpoint.get("input_docs", [])
                        if isinstance(checkpoint.get("input_docs"), list)
                        else []
                    ),
                ]
                if str(value).strip()
            ),
        }
        if not self._packet_matches_build(packet, checkpoint, allowed_docs):
            return False
        packet["build_id"] = build_id
        self.store.write_json(packet_key, packet)
        return True

    def _relation_manifest_packet_ids(self, checkpoint: Mapping[str, object]) -> set[str]:
        """Return all source packets owned by the current build.

        The manifest is deliberately stored outside the bounded blackboard
        state.  It is rebuilt from packet records so a conductor restart or a
        lost worker session cannot silently reduce relation scope to the last
        packet that happened to remain in context.  Legacy packets without a
        ``build_id`` are adopted only when they belong to the checkpoint's
        input document; the association is persisted for subsequent retries.
        """
        build_id = str(checkpoint.get("build_id", "")).strip()
        if not build_id:
            return set()
        allowed_docs = {
            str(value).strip()
            for value in [
                checkpoint.get("doc_id"),
                *(
                    checkpoint.get("input_docs", [])
                    if isinstance(checkpoint.get("input_docs"), list)
                    else []
                ),
            ]
            if str(value).strip()
        }
        packet_ids: set[str] = set()
        for key in self.store.list_dir("source_packets", recursive=True):
            if not key.endswith(".json"):
                continue
            packet = self.store.read_json(key)
            if not isinstance(packet, dict):
                continue
            packet_id = str(packet.get("packet_id", "")).strip()
            if not packet_id or str(packet.get("status", "complete")) == "failed":
                continue
            packet_build_id = str(packet.get("build_id", "")).strip()
            if packet_build_id == build_id:
                packet_ids.add(packet_id)
                continue
            if packet_build_id or not self._packet_matches_build(packet, checkpoint, allowed_docs):
                continue
            # Rolling upgrades left packet records without build ownership.
            # Adopt them exactly once at the materialized checkpoint; this is
            # safe for empty builds and gives legacy packets a durable owner.
            packet["build_id"] = build_id
            self.store.write_json(key, packet)
            packet_ids.add(packet_id)

        manifest = {
            "version": 1,
            "build_id": build_id,
            "doc_id": str(checkpoint.get("doc_id", "")),
            "input_docs": checkpoint.get("input_docs", []),
            "packet_ids": sorted(packet_ids),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self.store.write_json(self._relation_manifest_key(build_id), manifest)
        return packet_ids

    def _all_source_packet_ids(self) -> set[str]:
        """Return every readable packet id for checkpoint-free recovery."""
        packet_ids: set[str] = set()
        for key in self.store.list_dir("source_packets", recursive=True):
            if not key.endswith(".json"):
                continue
            packet = self.store.read_json(key)
            if not isinstance(packet, dict) or str(packet.get("status", "complete")) == "failed":
                continue
            packet_id = str(packet.get("packet_id", "")).strip()
            if packet_id:
                packet_ids.add(packet_id)
        return packet_ids

    def _relation_scope(self, checkpoint: Mapping[str, object]) -> tuple[set[str], str, bool]:
        """Resolve relation input without trusting a missing or hollow checkpoint.

        A complete checkpoint with declared input documents uses build ownership.
        A missing checkpoint, or a checkpoint written without input identity, is
        a recovery case: relation planning scans the durable packet records and
        reports that fact to the conductor.  This keeps a stale/empty manifest
        from turning a real relation workload into a false zero-candidate result.
        """
        checkpoint_exists = bool(checkpoint.get("exists"))
        build_id = str(checkpoint.get("build_id", "")).strip()
        input_docs = checkpoint.get("input_docs")
        has_input_identity = isinstance(input_docs, list) and any(
            isinstance(value, str) and value.strip() for value in input_docs
        )
        if checkpoint_exists and build_id and has_input_identity:
            return self._relation_manifest_packet_ids(checkpoint), "build_manifest", True
        return self._all_source_packet_ids(), "packet_store_recovery", False

    def _write_relation_work_ledger(
        self,
        scope_id: str,
        *,
        scope_source: str,
        scope_trusted: bool,
        packet_ids: set[str],
        items: list[dict[str, object]],
    ) -> str:
        """Persist bounded relation work for workers and restart recovery."""
        key = self._relation_work_key(scope_id)
        self.store.write_json(
            key,
            {
                "version": 1,
                "scope_id": scope_id,
                "scope_source": scope_source,
                "scope_trusted": scope_trusted,
                "packet_ids": sorted(packet_ids),
                "items": items,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        return f"{self.store.root_uri}/{key}"

    def get_source_ledger(self) -> dict[str, object]:
        """Project source packets into the legacy coverage response shape.

        New packets are independent records under ``source_packets/``. The
        aggregate mapping is built in memory so incremental writes never
        read-modify-write one ever-growing JSON document. A legacy aggregate
        ledger is merged read-only during rolling upgrades.
        """
        legacy = self.store.read_json(self.store._key_of(self._source_ledger_path()))
        packets: dict[str, object] = {}
        coverage: dict[str, object] = {}
        if legacy is not None:
            legacy_packets = legacy.get("packets")
            legacy_coverage = legacy.get("coverage")
            if isinstance(legacy_packets, dict):
                packets.update((str(key), value) for key, value in legacy_packets.items())
            if isinstance(legacy_coverage, dict):
                coverage.update((str(key), value) for key, value in legacy_coverage.items())

        for key in self.store.list_dir("source_packets", recursive=False):
            if not key.endswith(".json"):
                continue
            document = self.store.read_json(key)
            if not isinstance(document, dict):
                continue
            packet_id = str(document.get("packet_id", "")).strip()
            if not packet_id:
                continue
            record = {name: value for name, value in document.items() if name != "packet"}
            record["packet_uri"] = f"{self.store.root_uri}/{key}"
            packets[packet_id] = record
            source_hashes = record.get("source_hashes")
            if not isinstance(source_hashes, dict):
                continue
            source_uris = record.get("source_uris")
            if not isinstance(source_uris, list):
                continue
            for source_uri in source_uris:
                if not isinstance(source_uri, str):
                    continue
                previous = coverage.get(source_uri)
                packet_ids = (
                    list(previous.get("packet_ids", [])) if isinstance(previous, dict) else []
                )
                normalized_ids = {str(value) for value in packet_ids}
                normalized_ids.add(packet_id)
                coverage[source_uri] = {
                    "packet_ids": sorted(normalized_ids),
                    "status": record.get("status", "complete"),
                    "source_hash": source_hashes.get(source_uri, ""),
                }
        return {
            "version": 2,
            "packets": dict(sorted(packets.items())),
            "coverage": dict(sorted(coverage.items())),
        }

    def register_no_entity_chapters(
        self,
        doc_id: str,
        packet_id: str,
        source_uris: list[str],
        ledger_key: str = "",
        build_id: str = "",
    ) -> dict[str, object]:
        """Register low-value chapters as covered with a placeholder packet.

        Low-value rootsections are pure boilerplate — prefaces, safety
        notices, tables of contents, blank/cover pages — that carry no entity
        candidates.  They are NOT spec sheets, parameter tables, technical
        specifications, or schematic/figure chapters: those carry model
        identity, threshold values, wiring/line numbers and connection
        evidence and must be read by an extraction worker.  Instead of making
        a worker read pure boilerplate in full just to conclude ``no_entity``,
        the conductor/file_operator can register those here — the packet
        records the source URIs and hash, and its (empty) body satisfies the
        ``packet_body_missing`` coverage gate.

        Args:
            doc_id: Document id owning the chapters.
            packet_id: Stable ASCII slug identifying this packet.
            source_uris: Raw chapter URIs to mark covered.
            ledger_key: Deprecated compatibility input; packet storage is
                always derived from ``packet_id``.

        Returns:
            The ledger record written for this packet.
        """
        placeholder_body: dict[str, object] = {
            "kind": "no_entity",
            "note": "Low-value rootsection registered without model read; raw URIs are wired into Device.system_chapters by file_operator.",
        }
        # build_id MUST match the value passed to plan_chapter_work, because
        # the prefilter marker path is derived from sha256(build_id + doc_id).
        # If the conductor omits build_id here, the checkpoint backfill may
        # resolve a DIFFERENT build_id than what plan_chapter_work received,
        # writing the prefilter marker to a plan_id path that plan_chapter_work
        # never reads — pre_filter_required stays true forever.
        normalized_build_id = build_id.strip()
        if not normalized_build_id:
            checkpoint = self.store.read_json("index/build_checkpoint.json")
            if isinstance(checkpoint, dict):
                checkpoint_build_id = str(checkpoint.get("build_id", "")).strip()
                checkpoint_docs = {
                    str(value).strip()
                    for value in [
                        checkpoint.get("doc_id"),
                        *(
                            checkpoint.get("input_docs", [])
                            if isinstance(checkpoint.get("input_docs", []), list)
                            else []
                        ),
                    ]
                    if str(value).strip()
                }
                if checkpoint_build_id and doc_id in checkpoint_docs:
                    normalized_build_id = checkpoint_build_id
        if not normalized_build_id:
            raise ValueError(
                "build_id is required for register_no_entity_chapters; pass "
                "the same build_id used in plan_chapter_work so the prefilter "
                "marker is written to the correct plan_id path",
            )
        plan_id = sha256(f"{normalized_build_id}\x1f{doc_id}".encode()).hexdigest()[:24]
        self.store.write_json(
            f"index/chapter_plans/{plan_id}.prefilter.json",
            {"done": True, "packet_id": packet_id, "count": len(source_uris)},
            durable=True,
        )
        return self.record_source_packet(
            packet_id=packet_id,
            doc_id=doc_id,
            source_uris=source_uris,
            status="complete",
            evidence_count=0,
            packet_body=placeholder_body,
            ledger_key=ledger_key,
            build_id=normalized_build_id,
        )

    def record_source_packet(
        self,
        packet_id: str,
        doc_id: str,
        source_uris: list[str],
        *,
        source_hash: str = "",
        status: str = "complete",
        evidence_count: int = 0,
        packet_body: PacketBody | None = None,
        ledger_key: str = "",
        source_contents: dict[str, str] | None = None,
        allow_same_snapshot_replace: bool = False,
        build_id: str = "",
    ) -> dict[str, object]:
        """Persist packet body and deterministic chapter coverage.

        Every packet is stored independently under ``source_packets/`` using
        its stable business ``packet_id``. Coverage is projected from those
        records instead of mutating a central aggregate ledger.

        ``source_contents`` maps external MCP source URIs to the text the
        agent fetched via MCP tools. The system hashes this caller-provided
        text instead of trying to re-read the URI locally.

        ``allow_same_snapshot_replace`` is reserved for an explicit
        same-task refinement workflow. It permits a stable packet id to
        replace its extraction result when the source snapshot is unchanged;
        ordinary packet callers retain the conflict guard by default.
        """
        if not packet_id.strip() or not doc_id.strip():
            raise ValueError("packet_id and doc_id must not be empty")
        if status not in {"complete", "partial", "failed"}:
            raise ValueError("source packet status must be complete, partial, or failed")
        if evidence_count < 0:
            raise ValueError("evidence_count must be non-negative")
        if isinstance(packet_body, PacketBody):
            packet_body = packet_body.model_dump(exclude_none=True)
        sources = sorted({
            self._canonicalize_local_raw_uri(uri.strip()) for uri in source_uris if uri.strip()
        })
        normalized_source_contents = {
            self._canonicalize_local_raw_uri(uri): content
            for uri, content in (source_contents or {}).items()
        }
        if not sources:
            raise ValueError("source_uris must not be empty")
        kinds = [
            classify_raw_source_uri(uri, raw_root_uri=self._raw_fs.root_uri) for uri in sources
        ]
        is_chapters = all(
            kind is RawSourceKind.MANUAL_CHAPTER and self._source_belongs_to_doc(uri, doc_id)
            for uri, kind in zip(sources, kinds, strict=True)
        )
        is_case_file = all(
            kind is RawSourceKind.CASE and self._source_belongs_to_doc(uri, doc_id)
            for uri, kind in zip(sources, kinds, strict=True)
        )
        is_cross_raw = all(
            kind is not None and not uri.startswith(self._raw_fs.root_uri + "/")
            for uri, kind in zip(sources, kinds, strict=True)
        )
        if not (is_chapters or is_case_file or is_cross_raw):
            raise ValueError(
                "source_uris must be raw chapter URIs, the case file for the selected doc_id, or an addressable cross-namespace Markdown raw resource URI",
            )
        missing: list[str] = []
        source_hashes: dict[str, str] = {}
        for uri in sources:
            provided_content = normalized_source_contents.get(uri)
            if provided_content:
                source_hashes[uri] = sha256(provided_content.encode("utf-8")).hexdigest()
                continue
            content_hash = self._source_hash(uri, refresh=True)
            if content_hash is not None:
                source_hashes[uri] = content_hash
                continue
            missing.append(uri)
        if missing:
            raise ValueError(
                f"Source packet contains unreadable or unavailable sources outside the current snapshot: {missing[:5]}",
            )
        computed_hash = sha256(
            "\n".join(f"{uri}\x1f{source_hashes[uri]}" for uri in sources).encode(),
        ).hexdigest()
        if source_hash and source_hash != computed_hash:
            raise ValueError("source_hash does not match the current chapter contents")
        extractor_config_hash = self._materialization_config_hash()
        record: dict[str, object] = {
            "packet_id": packet_id,
            "doc_id": doc_id,
            "source_uris": sources,
            "source_hash": computed_hash,
            "source_hashes": source_hashes,
            "extractor_config_hash": extractor_config_hash,
            "status": status,
            "evidence_count": evidence_count,
        }
        normalized_build_id = build_id.strip()
        if normalized_build_id:
            record["build_id"] = normalized_build_id
        checkpoint = self.store.read_json("index/build_checkpoint.json")
        if not normalized_build_id and isinstance(checkpoint, dict):
            checkpoint_build_id = str(checkpoint.get("build_id", "")).strip()
            checkpoint_docs = {
                str(value).strip()
                for value in [
                    checkpoint.get("doc_id"),
                    *(
                        checkpoint.get("input_docs", [])
                        if isinstance(checkpoint.get("input_docs"), list)
                        else []
                    ),
                ]
                if str(value).strip()
            }
            if checkpoint_build_id and doc_id in checkpoint_docs:
                record["build_id"] = checkpoint_build_id
        if ledger_key and not re.fullmatch(
            r"(?:member/source|source_packets)/[A-Za-z0-9_]+",
            ledger_key,
        ):
            raise ValueError("legacy ledger_key must contain an ASCII packet slug")
        packet_key = self._source_packet_key(packet_id)
        packet_document = {**record, "packet": packet_body}
        existing_packet = self.store.read_json(packet_key)
        existing_build_id = (
            str(existing_packet.get("build_id", "")).strip()
            if isinstance(existing_packet, dict)
            else ""
        )
        requested_build_id = str(record.get("build_id", "")).strip()
        comparable_existing = dict(existing_packet) if isinstance(existing_packet, dict) else None
        if comparable_existing is not None:
            comparable_existing.pop("build_id", None)
        comparable_packet = dict(packet_document)
        comparable_packet.pop("build_id", None)
        if (
            existing_packet is not None
            and comparable_existing != comparable_packet
            and str(existing_packet.get("source_hash", "")) == computed_hash
            and str(existing_packet.get("extractor_config_hash", "")) == extractor_config_hash
            and not (
                existing_build_id and requested_build_id and existing_build_id != requested_build_id
            )
            and not allow_same_snapshot_replace
        ):
            raise ValueError(
                f"Source packet has different content for an unchanged source snapshot: {packet_id}",
            )
        if existing_build_id and requested_build_id and existing_build_id != requested_build_id:
            # ``packet_id`` is the stable identity of a source unit, not of one
            # execution.  A later build of the same document must replace the
            # latest packet in place after re-reading the source; otherwise
            # incremental manuals can never advance past the first build.
            packet_document["supersedes_build_id"] = existing_build_id
            record["supersedes_build_id"] = existing_build_id
        # A packet id is a stable business identity. Changed raw input
        # replaces the old extraction result instead of creating an
        # ever-growing family of version-suffixed packet ids.
        # The worker completes its task immediately after this receipt. Do not
        # report success until a separate planner read can observe the packet,
        # otherwise the same chapter is dispatched again.
        self.store.write_json(packet_key, packet_document, durable=True)
        if self._log:
            self._log.source_packet_recorded(packet_id, doc_id, len(sources))
        record["packet_uri"] = f"{self.store.root_uri}/{packet_key}"
        return record

    @staticmethod
    def _materialization_receipt_key(build_id: str, entity_uri: str) -> str:
        """Return a bounded receipt path for one build/entity input."""
        build_slug = sha256(build_id.encode("utf-8")).hexdigest()[:16]
        entity_slug = sha256(entity_uri.encode("utf-8")).hexdigest()[:24]
        return f"index/materialization_receipts/{build_slug}/{entity_slug}.json"
