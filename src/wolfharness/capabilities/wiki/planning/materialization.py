"""Materialization planning and template batch execution."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
import logging

from wolfharness.capabilities.wiki.io.template_materializer import (
    assemble_template_entity,
    strip_device_prefix,
)
from wolfharness.capabilities.wiki.quality import extract_sections, parse_frontmatter
from wolfharness.capabilities.wiki.schema_loader import get_schema_version
from wolfharness.capabilities.wiki.storage import (
    LocalFS,
    LocalVikingFS,
)


logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING

from wolfharness.capabilities.wiki._helpers import (
    _entity_batch_limit,
    _materialization_task_byte_limit,
)


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def _core_section_filled(content: str, section_heading: str) -> bool:
    """Check if entity content has a body section with real (non-placeholder) content."""
    idx = content.find(section_heading)
    if idx < 0:
        return False
    after = content[idx + len(section_heading) :]
    # Stop at the next section heading
    next_section = after.find("\n## ")
    if next_section >= 0:
        after = after[:next_section]
    after = after.strip()
    if not after:
        return False
    # Treat placeholder text as empty
    return "待补充" not in after


def _has_substantive_content(content: str) -> bool:
    """Check if entity body has any real (non-placeholder) content beyond frontmatter.

    Returns True if any ``##`` section has text that isn't just a placeholder.
    This detects whether a previous LLM worker or manual edit has added
    real content to the entity, which should be preserved on re-ingestion
    rather than overwritten with a fresh template.
    """
    sections = extract_sections(content)
    for body_text in sections.values():
        stripped = body_text.strip()
        if stripped and "待补充" not in stripped:
            return True
    return False


class MaterializationMixin:
    """Materialization planning and template batch execution."""

    def _materialization_candidates(
        self,
        *,
        build_id: str = "",
        doc_id: str = "",
    ) -> dict[str, dict[str, object]]:
        """Return current candidate inputs grouped by canonical entity URI."""
        candidates: list[dict[str, object]] = []
        packet_keys = [
            key
            for key in self.store.list_dir("source_packets", recursive=False)
            if key.endswith(".json")
        ]
        for key in packet_keys:
            packet = self.store.read_json(key)
            if not isinstance(packet, dict):
                continue
            packet_id = str(packet.get("packet_id", "")).strip()
            packet_build_id = str(packet.get("build_id", "")).strip()
            packet_doc_id = str(packet.get("doc_id", "")).strip()
            # "legacy" is a degraded-mode fallback build_id used when the
            # conductor loses the original build_id.  In that case, do NOT
            # filter by build_id — scan all packets so materialization can
            # recover candidates from the original build's source packets.
            if build_id and build_id not in ("legacy", packet_build_id):
                continue
            if doc_id and not (
                self._doc_id_variants(packet_doc_id) & self._doc_id_variants(doc_id)
            ):
                continue
            if str(packet.get("status", "complete")) != "complete":
                continue
            body = packet.get("packet")
            if (
                not packet_id
                or not isinstance(body, dict)
                or body.get("kind") in {"bom_identity_plan", "no_entity"}
            ):
                continue
            raw_candidates = body.get("normalized_identity_candidates")
            if raw_candidates is None:
                raw_candidates = body.get("entity_candidates", body.get("entities", []))
            if not isinstance(raw_candidates, list):
                continue
            for candidate in raw_candidates:
                if not isinstance(candidate, dict):
                    continue
                concept = candidate.get("concept")
                class_name = candidate.get("class_name", candidate.get("class"))
                object_name = candidate.get("object_name", candidate.get("name"))
                if not isinstance(concept, str) or not concept.strip():
                    continue
                # For Symptom, class_name is optional (directory layout drops class level)
                needs_class = concept != "Symptom"
                if needs_class and (not isinstance(class_name, str) or not class_name.strip()):
                    continue
                if not isinstance(object_name, str) or not object_name.strip():
                    continue
                normalized_concept = concept.strip()
                cls_value = (
                    class_name.strip()
                    if isinstance(class_name, str) and class_name.strip()
                    else None
                )
                if normalized_concept not in self.store.CONCEPT_DIRS:
                    # A malformed candidate must not abort the whole planner
                    # and force the conductor back into manual re-planning.
                    # Keep the packet available for audit, but only schedule
                    # concepts the store can materialize deterministically.
                    continue
                uri = self.store.entity_uri(normalized_concept, cls_value, object_name.strip())
                raw_source_uris = packet.get("source_uris", [])
                source_uris = raw_source_uris if isinstance(raw_source_uris, list) else []
                candidates.append(
                    {
                        "packet_id": packet_id,
                        "concept": normalized_concept,
                        "class_name": cls_value or "",
                        "object_name": object_name.strip(),
                        "entity_uri": uri,
                        "write_set": [uri],
                        "source_hash": str(packet.get("source_hash", "")),
                        "extractor_config_hash": str(packet.get("extractor_config_hash", "")),
                        "source_uris": [
                            str(uri) for uri in source_uris if isinstance(uri, str) and uri.strip()
                        ],
                        "materialization_inputs": [
                            {
                                "packet_id": packet_id,
                                "source_hash": str(packet.get("source_hash", "")),
                                "extractor_config_hash": str(
                                    packet.get("extractor_config_hash", "")
                                ),
                            },
                        ],
                    },
                )

        unique: dict[str, dict[str, object]] = {}
        for candidate in candidates:
            uri = str(candidate["entity_uri"])
            existing = unique.get(uri)
            if existing is None:
                unique[uri] = candidate
                continue
            raw_packet_ids = existing.get("packet_ids", [existing["packet_id"]])
            packet_ids = (
                list(raw_packet_ids)
                if isinstance(raw_packet_ids, list)
                else [existing["packet_id"]]
            )
            packet_id = str(candidate["packet_id"])
            if packet_id not in packet_ids:
                packet_ids.append(packet_id)
            existing["packet_ids"] = packet_ids
            raw_inputs = existing.get("materialization_inputs", [])
            inputs = list(raw_inputs) if isinstance(raw_inputs, list) else []
            inputs.append(
                {
                    "packet_id": packet_id,
                    "source_hash": str(candidate.get("source_hash", "")),
                    "extractor_config_hash": str(candidate.get("extractor_config_hash", "")),
                },
            )
            existing["materialization_inputs"] = inputs
            raw_existing_sources = existing.get("source_uris", [])
            raw_candidate_sources = candidate.get("source_uris", [])
            existing_sources = (
                raw_existing_sources if isinstance(raw_existing_sources, list) else []
            )
            candidate_sources = (
                raw_candidate_sources if isinstance(raw_candidate_sources, list) else []
            )
            source_uris = [str(uri) for uri in existing_sources]
            source_uris.extend(str(uri) for uri in candidate_sources)
            existing["source_uris"] = list(dict.fromkeys(source_uris))

        for candidate in unique.values():
            raw_inputs = candidate.get("materialization_inputs", [])
            inputs = list(raw_inputs) if isinstance(raw_inputs, list) else []
            if not inputs:
                inputs = [
                    {
                        "packet_id": str(candidate["packet_id"]),
                        "source_hash": str(candidate.get("source_hash", "")),
                        "extractor_config_hash": str(candidate.get("extractor_config_hash", "")),
                    },
                ]
            candidate["materialization_inputs"] = sorted(
                inputs, key=lambda item: str(item["packet_id"])
            )
            candidate["materialization_input_hash"] = sha256(
                json.dumps(
                    {
                        "entity_uri": candidate["entity_uri"],
                        "inputs": candidate["materialization_inputs"],
                        "schema_version": get_schema_version(),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            ).hexdigest()
        return unique

    def _chapter_plan_pending_count(
        self, doc_id: str, build_id: str
    ) -> list[dict[str, str]] | None:
        """Return pending chapter details when a server plan exists.

        Each element is ``{"uri": ..., "packet_id": ..., "reason": ...}`` so
        callers can act on specific chapters instead of guessing from a count.
        """
        plan_id = sha256(f"{build_id}\x1f{doc_id}".encode()).hexdigest()[:24]
        persisted = self.store.read_json(f"index/chapter_plans/{plan_id}.json")
        if not isinstance(persisted, dict):
            return None
        chapters = persisted.get("chapters")
        if not isinstance(chapters, list):
            return None
        checkpoint = self.store.read_json("index/build_checkpoint.json")
        ownership_checkpoint: Mapping[str, object] = (
            checkpoint
            if isinstance(checkpoint, dict)
            else {"doc_id": doc_id, "input_docs": [doc_id]}
        )
        # Build a fallback URI→packet map from ALL complete packets, same as
        # plan_chapter_work's completion scan.  This catches packets whose
        # packet_id differs from the plan's expected one (e.g. re-registered
        # with a build_id after initial legacy registration).
        complete_uri_owners: dict[str, str] = {}
        for pkt_key in self.store.list_dir("source_packets", recursive=True):
            if not pkt_key.endswith(".json"):
                continue
            owning = self.store.read_json(pkt_key)
            if not isinstance(owning, dict) or str(owning.get("status", "")) != "complete":
                continue
            owning_build_id = str(owning.get("build_id", "")).strip()
            if owning_build_id and owning_build_id != build_id:
                continue
            raw_owning_uris = (
                owning.get("source_uris", []) if isinstance(owning.get("source_uris"), list) else []
            )
            for raw_uri in raw_owning_uris:
                if isinstance(raw_uri, str) and raw_uri not in complete_uri_owners:
                    complete_uri_owners[raw_uri] = pkt_key

        pending: list[dict[str, str]] = []
        for item in chapters:
            if not isinstance(item, dict):
                pending.append({"uri": "", "packet_id": "", "reason": "malformed plan entry"})
                continue
            packet_id = str(item.get("packet_id", ""))
            uri = str(item.get("uri", ""))
            packet_key = self._source_packet_key(packet_id) if packet_id else ""
            packet = self.store.read_json(packet_key) if packet_key else None
            completed = self._chapter_packet_complete_for_build(
                packet_key,
                packet if isinstance(packet, dict) else None,
                build_id=build_id,
                doc_id=doc_id,
                source_uri=uri,
                checkpoint=ownership_checkpoint,
            )
            if completed or uri in complete_uri_owners:
                continue
            if not isinstance(packet, dict):
                reason = "packet missing"
            elif str(packet.get("status", "")) != "complete":
                reason = f"status={packet.get('status', 'missing')}"
            elif uri not in (
                packet.get("source_uris") if isinstance(packet.get("source_uris"), list) else []
            ):
                reason = "source_uri not in packet"
            else:
                reason = f"build_id mismatch (packet={packet.get('build_id', '')})"
            pending.append({"uri": uri, "packet_id": packet_id, "reason": reason})
        return pending

    def plan_materialization_work(
        self,
        *,
        build_id: str = "",
        doc_id: str = "",
        active_entity_uris: list[str] | None = None,
        audit_profile: str = "manual",
        max_parallel_shards: int | None = None,
    ) -> dict[str, object]:
        """Plan only unconsumed, URI-disjoint 1B materialization inputs.

        ``active_entity_uris`` is the conductor's bounded live write-set view.
        Excluding those items lets it fill newly free worker slots without a
        fixed wave barrier or duplicate dispatch. ``max_parallel_shards``
        bounds each response so embedded candidates cannot overflow the
        conductor context; omitted calls use a conservative first-wave limit.
        """
        if max_parallel_shards is not None and max_parallel_shards < 1:
            raise ValueError("max_parallel_shards must be positive when provided")
        response_shard_limit = min(max_parallel_shards or 50, 50)
        checkpoint = self.store.read_json("index/build_checkpoint.json")
        if isinstance(checkpoint, dict):
            build_id = build_id.strip() or str(checkpoint.get("build_id", "")).strip()
            doc_id = doc_id.strip() or str(checkpoint.get("doc_id", "")).strip()
        if audit_profile not in {"manual", "case"}:
            raise ValueError("audit_profile must be manual or case")
        if build_id and doc_id:
            pending_chapters = self._chapter_plan_pending_count(doc_id, build_id)
            if pending_chapters:
                detail = "; ".join(f"{p['uri']} ({p['reason']})" for p in pending_chapters[:10])
                raise ValueError(
                    f"Chapter source-packet coverage is incomplete; "
                    f"finish pending chapter work before materialization. "
                    f"pending_count={len(pending_chapters)}, pending: {detail}",
                )

        all_candidates = self._materialization_candidates(build_id=build_id, doc_id=doc_id)
        # In legacy recovery mode, receipts may have been written under the
        # original build_id.  Try both "legacy" and any build_id found on the
        # candidates' source packets to avoid re-materializing already-done
        # entities.
        receipt_build_ids = [build_id or "legacy"]
        if build_id == "legacy":
            for candidate in all_candidates.values():
                pid = str(candidate.get("packet_id", ""))
                if pid:
                    pkt = self.store.read_json(self._source_packet_key(pid))
                    if isinstance(pkt, dict):
                        bid = str(pkt.get("build_id", "")).strip()
                        if bid and bid not in receipt_build_ids:
                            receipt_build_ids.append(bid)
                    break  # one packet is enough to discover the original build_id
        unique: dict[str, dict[str, object]] = {}
        completed_count = 0
        for uri, candidate in all_candidates.items():
            already_done = False
            for rbid in receipt_build_ids:
                receipt = self.store.read_json(self._materialization_receipt_key(rbid, uri))
                if (
                    isinstance(receipt, dict)
                    and str(receipt.get("materialization_input_hash", ""))
                    == str(candidate.get("materialization_input_hash", ""))
                    and self.store.read_entity_by_uri(str(receipt.get("result_entity_uri", uri)))
                    is not None
                ):
                    already_done = True
                    break
            if already_done:
                completed_count += 1
                continue
            unique[uri] = candidate

        active = {
            uri.strip()
            for uri in (active_entity_uris or [])
            if uri.strip() and uri.strip() in unique
        }
        dispatchable = {uri: candidate for uri, candidate in unique.items() if uri not in active}

        grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
        for candidate in dispatchable.values():
            concept = str(candidate["concept"])
            # Symptom no longer partitions by class_name (directory layout
            # drops the class level), so all Symptoms share one partition —
            # unrelated URI sets still run concurrently.
            class_partition = ""
            grouped.setdefault((concept, class_partition), []).append(candidate)
        shards: list[dict[str, object]] = []
        shard_index = 0
        packet_payload_cache: dict[str, dict[str, object]] = {}

        def packet_payload(packet_id: str) -> dict[str, object]:
            cached = packet_payload_cache.get(packet_id)
            if cached is not None:
                return cached
            packet = self.store.read_json(self._source_packet_key(packet_id))
            payload = {
                "source_uris": packet.get("source_uris", []) if isinstance(packet, dict) else [],
                "packet": packet.get("packet", {}) if isinstance(packet, dict) else {},
            }
            packet_payload_cache[packet_id] = payload
            return payload

        def materialization_chunks(items: list[dict[str, object]]) -> list[list[dict[str, object]]]:
            chunks: list[list[dict[str, object]]] = []
            current: list[dict[str, object]] = []
            for item in items:
                trial = [*current, item]
                packet_ids: set[str] = set()
                for candidate in trial:
                    raw_packet_ids = candidate.get("packet_ids", [candidate["packet_id"]])
                    candidate_packet_ids = (
                        raw_packet_ids
                        if isinstance(raw_packet_ids, list)
                        else [candidate["packet_id"]]
                    )
                    packet_ids.update(
                        str(packet_id)
                        for packet_id in candidate_packet_ids
                        if str(packet_id).strip()
                    )
                estimated_payload = {
                    "entity_candidates": trial,
                    "source_packet_payloads": {
                        packet_id: packet_payload(packet_id) for packet_id in sorted(packet_ids)
                    },
                }
                exceeds_count = len(trial) > _entity_batch_limit()
                exceeds_bytes = (
                    len(
                        json.dumps(
                            estimated_payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8"),
                    )
                    > _materialization_task_byte_limit()
                )
                if current and (exceeds_count or exceeds_bytes):
                    chunks.append(current)
                    current = [item]
                else:
                    current = trial
            if current:
                chunks.append(current)
            return chunks

        def _is_single_packet(candidate: dict[str, object]) -> bool:
            # A candidate is multi-packet when its aggregated identity spans
            # more than one source packet (cross-packet merge).  Such
            # candidates need LLM reconciliation; single-packet ones can be
            # rendered from packet fields alone (template path, zero LLM).
            pids = candidate.get("packet_ids", [candidate["packet_id"]])
            return not (isinstance(pids, list) and len(pids) > 1)

        # ponytail: removed packet_concept_counts guard — template now writes
        # a draft for ALL single-packet candidates, even when a packet
        # produces multiple same-concept candidates.  The template body will
        # be identical across siblings (packet-level fields), but the entity
        # exists immediately with correct identity/frontmatter.
        # materialize_template_batch detects shared-packet candidates and
        # adds them to fallback_to_llm (with written=True) so the conductor
        # dispatches LLM workers to patch/differentiate the body in parallel.
        # Net effect: entities exist in seconds instead of 5-10 min/shard,
        # and LLM workers do lighter patch work instead of full creation.

        def _needs_llm(candidate: dict[str, object]) -> bool:
            """Whether a candidate must go through the LLM materialization path."""
            return not _is_single_packet(candidate)

        for concept, class_partition in sorted(grouped):
            items = sorted(
                grouped[(concept, class_partition)], key=lambda item: str(item["entity_uri"])
            )
            # Split template-eligible from LLM-requiring BEFORE chunking.
            # Multi-packet merges need LLM reconciliation; single-packet
            # candidates that share their packet with other same-concept
            # candidates also need LLM (template renders them identically).
            for items_subset, subset_strategy in (
                ([it for it in items if not _needs_llm(it)], "template"),
                ([it for it in items if _needs_llm(it)], "llm"),
            ):
                for chunk in materialization_chunks(items_subset):
                    shard_index += 1
                    uris = [str(item["entity_uri"]) for item in chunk]
                    packet_id_set: set[str] = set()
                    for item in chunk:
                        raw_packet_ids = item.get("packet_ids", [item["packet_id"]])
                        item_packet_ids = (
                            raw_packet_ids
                            if isinstance(raw_packet_ids, list)
                            else [item["packet_id"]]
                        )
                        packet_id_set.update(
                            str(packet_id)
                            for packet_id in item_packet_ids
                            if str(packet_id).strip()
                        )
                    packet_ids = sorted(packet_id_set)
                    shard_id = f"materialize_{concept.casefold()}_{shard_index}"
                    scope = (
                        "materialize:" + sha256("\n".join(uris).encode("utf-8")).hexdigest()[:16]
                    )
                    strategy = subset_strategy
                    if strategy == "template":
                        description = "\n".join(
                            [
                                "worker_role: wiki_extraction_worker",
                                "phase=1B_materialization",
                                f"build_id={build_id or 'legacy'}",
                                f"doc_id={doc_id}",
                                f"packet_id={packet_ids[0]}",
                                f"packet_ids={json.dumps(packet_ids, ensure_ascii=False, separators=(',', ':'))}",
                                f"entity_type={concept}",
                                f"audit_profile={audit_profile}",
                                "depends_on_stage=1B_materialization",
                                "expected_artifacts=materialized_entities",
                                f"shard_id={shard_id}",
                                f"chunk_id={shard_id}",
                                "chunk_of=1",
                                f"write_scope={scope}",
                                f"write_set={json.dumps(uris, ensure_ascii=False, separators=(',', ':'))}",
                                "materialization_strategy=template",
                                "materialization_mode=embedded_candidates_only",
                                "source_packet_reread=forbidden",
                                "raw_chapter_reread=forbidden",
                                "Call materialize_template_batch(packet_ids=<packet_ids>, entity_uris=<write_set>, build_id=<build_id>, doc_id=<doc_id>) directly. Do NOT dispatch a worker. The tool reads packets server-side, assembles entity bodies from packet fields, and writes them. Check fallback_to_llm in the result; dispatch workers only for those.",
                                "entity_candidates="
                                + json.dumps(chunk, ensure_ascii=False, separators=(",", ":")),
                            ],
                        )
                    else:
                        is_local_backend = isinstance(self.store._fs, (LocalFS, LocalVikingFS))
                        llm_lines = [
                            "worker_role: wiki_extraction_worker",
                            "phase=1B_materialization",
                            f"build_id={build_id or 'legacy'}",
                            f"doc_id={doc_id}",
                            f"packet_id={packet_ids[0]}",
                            f"packet_ids={json.dumps(packet_ids, ensure_ascii=False, separators=(',', ':'))}",
                            f"entity_type={concept}",
                            f"audit_profile={audit_profile}",
                            "depends_on_stage=1B_materialization",
                            "expected_artifacts=materialized_entities",
                            f"shard_id={shard_id}",
                            f"chunk_id={shard_id}",
                            "chunk_of=1",
                            f"write_scope={scope}",
                            f"write_set={json.dumps(uris, ensure_ascii=False, separators=(',', ':'))}",
                            "materialization_strategy=llm",
                        ]
                        if is_local_backend:
                            llm_lines.extend([
                                "materialization_mode=local_packet_reference",
                                "source_packet_reread=allowed",
                                "raw_chapter_reread=forbidden",
                                "Build is local: call record_source_packet to register evidence. Do NOT embed payloads.",
                                "entity_candidates="
                                + json.dumps(chunk, ensure_ascii=False, separators=(",", ":")),
                            ])
                        else:
                            source_packet_payloads = {
                                packet_id: packet_payload(packet_id) for packet_id in packet_ids
                            }
                            llm_lines.extend([
                                "materialization_mode=embedded_candidates_only",
                                "source_packet_reread=forbidden",
                                "raw_chapter_reread=forbidden",
                                "Use source_packet_payloads and entity_candidates embedded below; do not decode the source again.",
                                "source_packet_payloads="
                                + json.dumps(
                                    source_packet_payloads,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                                "entity_candidates="
                                + json.dumps(chunk, ensure_ascii=False, separators=(",", ":")),
                            ])
                        llm_lines.append(
                            "existing_entity_check=mandatory: Before writing any entity, "
                            "call list_children(<wiki_root>/<Concept>) to check if the entity "
                            "already exists. If it exists, use patch_entity (with expected_sha256 "
                            "from diff_entity) to merge — do NOT blind-write with write_entity. "
                            "If facts conflict, create_opa instead of overwriting.",
                        )
                        description = "\n".join(llm_lines)
                    shards.append(
                        {
                            "shard_id": shard_id,
                            "concept": concept,
                            "packet_ids": packet_ids,
                            "entity_uris": uris,
                            "write_set": uris,
                            "write_scope": scope,
                            "candidate_count": len(chunk),
                            "strategy": strategy,
                            "task_description": description,
                        },
                    )
        returned_candidate_count = sum(
            int(shard["candidate_count"]) for shard in shards[:response_shard_limit]
        )
        dispatchable_shards = shards
        shards = dispatchable_shards[:response_shard_limit]
        return {
            "scope_source": "source_packets",
            "build_id": build_id,
            "doc_id": doc_id,
            "shards": shards,
            "shard_count": len(shards),
            "candidate_count": returned_candidate_count,
            "dispatchable_shard_count": len(dispatchable_shards),
            "remaining_shard_count": len(dispatchable_shards) - len(shards),
            "pending_count": len(unique),
            "in_flight_count": len(active),
            "total_candidate_count": len(all_candidates),
            "completed_candidate_count": completed_count,
            "done": not unique,
            "idempotent": True,
        }

    def record_materialization_receipt(
        self,
        build_id: str,
        entity_uris: list[str],
        resolved_entity_uris: dict[str, str] | None = None,
    ) -> dict[str, object]:
        """Commit receipts after a shard writes its planned entity identities.

        ``resolved_entity_uris`` explicitly records a source-backed identity
        normalization (planned URI -> actual URI). The receipt remains keyed
        by the planned input, so normalization cannot make the planner
        dispatch the same candidate forever.
        """
        normalized_build_id = build_id.strip()
        if not normalized_build_id:
            raise ValueError("build_id is required")
        selected = list(dict.fromkeys(uri.strip() for uri in entity_uris if uri.strip()))
        if not selected:
            raise ValueError("entity_uris must not be empty")
        candidates = self._materialization_candidates(build_id=normalized_build_id)
        resolved = {
            planned.strip(): actual.strip()
            for planned, actual in (resolved_entity_uris or {}).items()
            if planned.strip() and actual.strip()
        }
        writes: list[tuple[str, str]] = []
        for uri in selected:
            candidate = candidates.get(uri)
            if candidate is None:
                raise ValueError(f"Entity is not a current materialization candidate: {uri}")
            result_uri = resolved.get(uri, uri)
            result_identity = self.store.lookup_by_uri(result_uri)
            if result_identity is None or str(result_identity[0]) != str(candidate["concept"]):
                raise ValueError(
                    f"Resolved materialization URI must exist and retain the planned concept: {uri} -> {result_uri}",
                )
            content = self.store.read_entity_by_uri(result_uri)
            if content is None:
                raise ValueError(f"Materialized entity does not exist: {result_uri}")
            record = {
                "version": 2,
                "build_id": normalized_build_id,
                "entity_uri": uri,
                "result_entity_uri": result_uri,
                "packet_ids": candidate.get("packet_ids", [candidate.get("packet_id", "")]),
                "materialization_input_hash": candidate["materialization_input_hash"],
                "result_entity_sha256": sha256(content.encode("utf-8")).hexdigest(),
                "recorded_at": datetime.now(UTC).isoformat(),
            }
            writes.append(
                (
                    self._materialization_receipt_key(normalized_build_id, uri),
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                ),
            )
        self.store.commit_many(writes)
        return {
            "build_id": normalized_build_id,
            "recorded_count": len(writes),
            "entity_uris": selected,
            "resolved_entity_uris": {uri: resolved[uri] for uri in selected if uri in resolved},
        }

    # ── Chapter map (title ↔ URI, enables direct TOC-title-based lookup) ──────

    def materialize_template_batch(
        self,
        *,
        packet_ids: list[str],
        entity_uris: list[str],
        build_id: str = "",
        doc_id: str = "",
    ) -> dict[str, object]:
        """Materialize entities from packet fields without an LLM worker.

        Fast path for single-packet candidates: reads the source packet body,
        assembles a draft entity page from ``source_subject`` /
        ``explicit_facts`` / ``parts_and_specs`` / ``evidence_map`` /
        ``ordered_actions``, and writes it via ``write_entities_batch``.

        Entities whose object_name carries a device-model prefix (e.g.
        ``SY75后处理DPF``) are automatically normalized by stripping the
        prefix; the planned URI → actual URI mapping is recorded via
        ``resolved_entity_uris`` in the materialization receipt.

        Entities that fail validation (e.g. irreparable object_name) are
        returned in ``fallback_to_llm`` so the conductor can dispatch them to
        a worker instead.
        """
        normalized_build_id = build_id.strip()
        if not normalized_build_id:
            checkpoint = self.store.read_json("index/build_checkpoint.json")
            if isinstance(checkpoint, dict):
                normalized_build_id = str(checkpoint.get("build_id", "")).strip()
                doc_id = doc_id.strip() or str(checkpoint.get("doc_id", "")).strip()
        if not normalized_build_id:
            raise ValueError("build_id is required (pass explicitly or set checkpoint)")

        checkpoint = self.store.read_json("index/build_checkpoint.json")
        device_id = (
            str(checkpoint.get("device_id", "")).strip() if isinstance(checkpoint, dict) else ""
        )
        series_id = (
            str(checkpoint.get("series_id", "")).strip() if isinstance(checkpoint, dict) else ""
        )
        device_model = device_id or series_id

        candidates = self._materialization_candidates(build_id=normalized_build_id, doc_id=doc_id)
        requested = list(dict.fromkeys(uri.strip() for uri in entity_uris if uri.strip()))
        if not requested:
            raise ValueError("entity_uris must not be empty")

        batch_limit = _entity_batch_limit()
        written_uris: list[str] = []
        resolved_map: dict[str, str] = {}
        fallback_to_llm: list[dict[str, str]] = []
        errors: list[dict[str, str]] = []
        skipped_existing = 0

        # Group by packet for efficient reads; only single-packet candidates
        # are template-eligible.  Multi-packet (merge) candidates go to fallback.
        pending: list[dict[str, object]] = []
        for uri in requested:
            candidate = candidates.get(uri)
            if candidate is None:
                fallback_to_llm.append({
                    "entity_uri": uri,
                    "reason": "not a current materialization candidate",
                })
                continue
            raw_packet_ids = candidate.get("packet_ids", [candidate.get("packet_id", "")])
            pid_list = (
                raw_packet_ids
                if isinstance(raw_packet_ids, list)
                else [str(candidate.get("packet_id", ""))]
            )
            if len(pid_list) > 1:
                fallback_to_llm.append({
                    "entity_uri": uri,
                    "reason": "multi-packet merge requires LLM",
                })
                continue
            pending.append(candidate)

        # Process in batches respecting the entity write limit.
        for batch_start in range(0, len(pending), batch_limit):
            batch = pending[batch_start : batch_start + batch_limit]
            entities_to_write: list[dict[str, object]] = []
            batch_meta: list[
                tuple[str, str, str, str, str]
            ] = []  # (planned_uri, concept, class_name, actual_object_name, packet_id)

            for candidate in batch:
                concept = str(candidate["concept"])
                class_name = str(candidate["class_name"])
                original_object_name = str(candidate["object_name"])
                planned_uri = str(candidate["entity_uri"])
                packet_id = str(candidate.get("packet_id", ""))

                packet = self.store.read_json(self._source_packet_key(packet_id))
                if not isinstance(packet, dict):
                    fallback_to_llm.append({
                        "entity_uri": planned_uri,
                        "reason": f"packet not readable: {packet_id}",
                    })
                    continue
                body = packet.get("packet")
                if not isinstance(body, dict):
                    fallback_to_llm.append({
                        "entity_uri": planned_uri,
                        "reason": f"packet body missing: {packet_id}",
                    })
                    continue

                # Component/Fault with empty mechanism text → LLM inference needed
                if concept == "Component":
                    if not str(body.get("working_mechanism", "")).strip():
                        fallback_to_llm.append({
                            "entity_uri": planned_uri,
                            "reason": "empty working_mechanism requires LLM inference",
                        })
                        continue
                elif concept == "Fault":
                    if not str(body.get("failure_mechanism", "")).strip():
                        fallback_to_llm.append({
                            "entity_uri": planned_uri,
                            "reason": "empty failure_mechanism requires LLM inference",
                        })
                        continue

                # Strip device-model prefix for Component (validation rule).
                actual_object_name = original_object_name
                if concept == "Component" and device_model:
                    actual_object_name = strip_device_prefix(
                        original_object_name, device_id, series_id
                    )

                content = assemble_template_entity(
                    packet_body=body,
                    concept=concept,
                    class_name=class_name,
                    object_name=actual_object_name,
                    device_model=device_model,
                    source_uris=list(candidate.get("source_uris", [])),  # type: ignore[arg-type]
                )

                # Incremental fill-blanks: skip or route entities based on
                # existing library content to avoid blind overwrites on re-ingestion.
                existing = self.store.read_entity(concept, class_name or None, actual_object_name)
                expected_sha = ""
                if existing is not None:
                    existing_fm = parse_frontmatter(existing)
                    existing_status = str(existing_fm.get("status", "")).strip()
                    if existing_status in ("confirmed", "published"):
                        skipped_existing += 1
                        continue
                    # Content equivalence: skip if body sections are identical
                    # (frontmatter may differ due to timestamps or source hashes).
                    if extract_sections(content) == extract_sections(existing):
                        skipped_existing += 1
                        continue
                    # Check if the core section already has real content
                    core_section = (
                        "## 工作机理"
                        if concept == "Component"
                        else "## 失效机理"
                        if concept == "Fault"
                        else None
                    )
                    if core_section and _core_section_filled(existing, core_section):
                        skipped_existing += 1
                        continue
                    # Route to LLM merge if existing entity has any substantive
                    # content that would be lost by a template overwrite.
                    if _has_substantive_content(existing):
                        fallback_to_llm.append({
                            "entity_uri": planned_uri,
                            "reason": "existing entity has content requiring LLM merge with new packet",
                            "written": False,
                        })
                        continue
                    expected_sha = sha256(existing.encode("utf-8")).hexdigest()

                entities_to_write.append({
                    "concept": concept,
                    "class_name": class_name,
                    "object_name": actual_object_name,
                    "content": content,
                    "expected_sha256": expected_sha,
                })
                batch_meta.append((planned_uri, concept, class_name, actual_object_name, packet_id))

            if not entities_to_write:
                continue

            try:
                write_result = self.write_entities_batch(entities_to_write)
                raw_written = write_result.get("uris", [])
                written_list: list[str] = (
                    [str(u) for u in raw_written] if isinstance(raw_written, list) else []
                )
                for idx, (planned_uri, _concept, _cls, _obj, _pid) in enumerate(batch_meta):
                    actual_uri = written_list[idx] if idx < len(written_list) else planned_uri
                    if actual_uri != planned_uri:
                        resolved_map[planned_uri] = actual_uri
                    written_uris.append(planned_uri)
                # Create Symptom Profile skeletons with resolved sibling URIs.
                # A plain skeleton (draft + placeholders) needs no LLM; only
                # skeleton creation failures fall back to the LLM worker.
                for planned_uri, _concept, _cls, _obj, packet_id in batch_meta:
                    if _concept != "Symptom":
                        continue
                    try:
                        symptom_uri = resolved_map.get(planned_uri, planned_uri)
                        packet = self.store.read_json(self._source_packet_key(packet_id))
                        if not isinstance(packet, dict):
                            raise ValueError(f"packet not readable: {packet_id}")
                        raw_sources = packet.get("source_uris", [])
                        source_uris = [
                            str(uri) for uri in raw_sources if isinstance(uri, str) and uri.strip()
                        ]
                        component = next(
                            (
                                entry
                                for entry in batch_meta
                                if entry[4] == packet_id and entry[1] == "Component"
                            ),
                            None,
                        )
                        if component is None:
                            raise ValueError(
                                f"no sibling Component in packet {packet_id} for profile skeleton",
                            )
                        fault_uris = [
                            self.store.entity_uri(entry[1], entry[2] or None, entry[3])
                            for entry in batch_meta
                            if entry[4] == packet_id and entry[1] == "Fault"
                        ]
                        device_uri = ""
                        if device_model:
                            try:
                                device_uri = self.store.entity_uri("Device", None, device_model)
                            except ValueError:
                                device_uri = ""
                        if not device_uri:
                            device_entities = self.store.list_entities("Device")
                            if device_entities:
                                device_uri = device_entities[0][3]
                        profile_id = (
                            f"{device_model}_{component[3]}"
                            .lower()
                            .replace(" ", "_")
                            .replace("/", "_")
                            .replace("\\", "_")
                            .replace("／", "_")
                            .replace("＼", "_")[:80]
                        )
                        content = self._symptom_profile_skeleton(
                            profile_id=profile_id,
                            symptom_uri=symptom_uri,
                            device_model=device_model,
                            component_name=component[3],
                            component_uri=self.store.entity_uri(
                                component[1], component[2] or None, component[3]
                            ),
                            fault_uris=fault_uris,
                            device_uri=device_uri,
                            source_uris=source_uris,
                        )
                        self.write_symptom_profile(symptom_uri, profile_id, content)
                    except (ValueError, TypeError, KeyError) as exc:
                        fallback_to_llm.append({
                            "entity_uri": planned_uri,
                            "reason": f"symptom profile creation requires LLM: {exc}",
                        })
            except (ValueError, TypeError) as exc:
                # Validation rejected the write — retry per-entity, fallback to LLM only where the single write also fails.
                for idx, (planned_uri, _concept, _cls, _obj, _pid) in enumerate(batch_meta):
                    raw_single = entities_to_write[idx] if idx < len(entities_to_write) else None
                    if isinstance(raw_single, list):
                        try:
                            self.write_entities_batch(raw_single)
                            continue
                        except (ValueError, TypeError) as single_exc:
                            fallback_to_llm.append({
                                "entity_uri": planned_uri,
                                "reason": f"per-entity retry: {single_exc}",
                            })
                            continue
                    fallback_to_llm.append({
                        "entity_uri": planned_uri,
                        "reason": f"batch validation: {exc}",
                    })

        # Record materialization receipts for successfully written entities.
        recorded_count = 0
        if written_uris:
            try:
                receipt = self.record_materialization_receipt(
                    build_id=normalized_build_id,
                    entity_uris=written_uris,
                    resolved_entity_uris=resolved_map or None,
                )
                recorded_count = int(str(receipt.get("recorded_count", 0)))
            except (ValueError, KeyError) as exc:
                errors.append({"receipt_error": str(exc)})

        # Detect shared-packet candidates: multiple same-concept entities from
        # one packet get identical template bodies.  Flag them for LLM body
        # differentiation — the entity IS written (draft), so the worker
        # patches rather than creates from scratch.
        if written_uris:
            packet_concept_counts: dict[tuple[str, str], int] = {}
            for candidate in candidates.values():
                raw_pids = candidate.get("packet_ids", [candidate.get("packet_id", "")])
                pid_list = (
                    raw_pids
                    if isinstance(raw_pids, list)
                    else [str(candidate.get("packet_id", ""))]
                )
                if len(pid_list) != 1:
                    continue
                pid = str(pid_list[0]).strip()
                if not pid:
                    continue
                key = (str(candidate["concept"]), pid)
                packet_concept_counts[key] = packet_concept_counts.get(key, 0) + 1

            for uri in written_uris:
                candidate = candidates.get(uri)
                if candidate is None:
                    continue
                raw_pids = candidate.get("packet_ids", [candidate.get("packet_id", "")])
                pid_list = (
                    raw_pids
                    if isinstance(raw_pids, list)
                    else [str(candidate.get("packet_id", ""))]
                )
                if len(pid_list) != 1:
                    continue
                pid = str(pid_list[0]).strip()
                if not pid:
                    continue
                key = (str(candidate["concept"]), pid)
                if packet_concept_counts.get(key, 0) > 1:
                    fallback_to_llm.append({
                        "entity_uri": uri,
                        "reason": "shared-packet candidate needs LLM body differentiation",
                        "written": True,
                    })

        return {
            "build_id": normalized_build_id,
            "written_count": len(written_uris),
            "recorded_count": recorded_count,
            "fallback_count": len(fallback_to_llm),
            "fallback_to_llm": fallback_to_llm,
            "resolved_entity_uris": resolved_map,
            "skipped_existing": skipped_existing,
            "errors": errors,
        }

    def _symptom_profile_skeleton(
        self,
        *,
        profile_id: str,
        symptom_uri: str,
        device_model: str,
        component_name: str,
        component_uri: str,
        fault_uris: list[str],
        device_uri: str,
        source_uris: list[str],
    ) -> str:
        """Build a draft Symptom Profile skeleton with resolved sibling URIs.

        Body sections carry LLM placeholders; a later worker phase fills them
        with content extracted from the source chapters.
        """
        fault_block = "".join(f"  - {uri}\n" for uri in fault_uris) or "  []\n"
        source_block = "".join(f"  - {uri}\n" for uri in source_uris) or "  []\n"
        return (
            "---\n"
            f"id: PRO-{profile_id}\n"
            f"profile_id: {profile_id}\n"
            f"parent_symptom: {symptom_uri}\n"
            "status: draft\n"
            "applicable_models:\n"
            f"  - {device_model}\n"
            "device_refs:\n"
            f"  - {device_uri}\n"
            f"direct_component_uri: {component_uri}\n"
            "possible_faults:\n"
            f"{fault_block}"
            "sources:\n"
            f"{source_block}"
            "---\n"
            "\n"
            "## 适用配置\n"
            "\n"
            f"{device_model} + {component_name}\n"
            "\n"
            "## 表现特征\n"
            "\n"
            "> 待 LLM 补充：从源章节提取该设备+Component组合下的具体表现特征。\n"
            "\n"
            "## 差异\n"
            "\n"
            "> 待 LLM 补充：与其他设备/配置组合的差异。\n"
            "\n"
            "## 可能失效机理\n"
            "\n"
            "> 待 LLM 补充：从源章节提取可能的失效机理。\n"
            "\n"
            "## 推荐诊断流程\n"
            "\n"
            "> 待 LLM 补充：从源章节提取推荐诊断流程。\n"
            "\n"
            "## 来源\n"
            "\n" + "".join(f"[{uri}]({uri})\n" for uri in source_uris)
        )

    def _source_belongs_to_doc(self, uri: str, doc_id: str) -> bool:
        """Return whether a raw URI belongs to the selected logical document.

        Parsed manuals normally live at ``<root>/<doc_id>/chapters``. Some
        Viking imports preserve the source tree under an extra catalog
        directory (for example ``menu/<display-doc-id>/...``). The discovered
        prefix is authoritative when available; the segment fallback keeps
        packet recording usable before the chapter map is persisted.
        """
        root_prefix = self._raw_fs.root_uri.rstrip("/") + "/"
        if not uri.startswith(root_prefix):
            return False
        relative = uri.removeprefix(root_prefix).strip("/")
        logical_doc = doc_id.strip("/")
        relative_folded = relative.casefold()
        logical_doc_folded = logical_doc.casefold()
        if logical_doc_folded and (
            relative_folded == logical_doc_folded
            or relative_folded.startswith(logical_doc_folded + "/")
        ):
            return True
        prefix = self._raw_doc_prefixes.get(doc_id, "")
        # Empty prefix means the document IS the raw root (library root is the
        # document); every raw URI under root belongs to it.
        if doc_id in self._raw_doc_prefixes and not prefix.strip():
            return True
        if prefix and relative.startswith(prefix.strip("/") + "/"):
            return True
        expected = logical_doc_folded
        return any(part.casefold() == expected for part in relative.split("/")[:-1])

    def _discover_nonstandard_doc_files(self, doc_id: str) -> tuple[str, list[str]]:
        """Find a logical document nested below a catalog directory.

        The remote raw backend may not provide recursive glob, so this walks
        only directory listings and stops the search once a directory segment
        matches the requested document id. It returns the matched prefix and
        all markdown leaves below it.
        """
        expected = doc_id.casefold()
        visited: set[str] = set()

        def collect(prefix: str) -> list[str]:
            leaves: list[str] = []
            pending = [prefix]
            while pending:
                current = pending.pop()
                if current in visited:
                    continue
                visited.add(current)
                for entry in self._raw_fs.list_entries(current, recursive=False):
                    if self._raw_fs.is_dir(entry):
                        pending.append(entry)
                    elif entry.endswith(".md"):
                        leaves.append(entry)
            return sorted(leaves)

        def search(prefix: str) -> tuple[str, list[str]] | None:
            for entry in self._raw_fs.list_entries(prefix, recursive=False):
                if not self._raw_fs.is_dir(entry):
                    continue
                name = entry.rsplit("/", 1)[-1]
                if name.casefold() == expected:
                    return entry, collect(entry)
                found = search(entry)
                if found is not None:
                    return found
            return None

        found = search("")
        return found if found is not None else ("", [])

    def _chapter_map_path(self) -> Path:
        """Path to the persisted chapter map JSON file."""
        return self.store.root / "chapter_map.json"

    def _save_chapter_map(self, doc_id: str, chapters: list[dict]) -> None:
        """Build and persist title → raw chapter URI mapping for a document."""
        mapping = self._build_chapter_map(doc_id, chapters, self.make_source_uri)
        self._chapter_map_cache[doc_id] = mapping
        map_path = self._chapter_map_path()
        all_maps: dict[str, dict[str, str]] = {}
        raw = self.store.read_text(self.store._key_of(map_path))
        if raw is not None:
            all_maps = json.loads(raw)
        all_maps[doc_id] = mapping
        self.store.write_json(self.store._key_of(map_path), all_maps)

    def read_chapter_map(self, doc_id: str) -> dict[str, str]:
        """Return the title → raw chapter URI mapping for a document.

        Keys are clean chapter titles (e.g. ``发动机冒黑烟``) or raw TOC
        titles.  Values are real-path raw URIs of the form
        ``{root_uri}/{doc_id}/chapters/<subdir>/chapter.md``.
        """
        cached = self._chapter_map_cache.get(doc_id)
        if cached is not None:
            return cached
        # Try persisted map first.
        map_path = self._chapter_map_path()
        raw = self.store.read_text(self.store._key_of(map_path))
        if raw is not None:
            all_maps: dict[str, dict[str, str]] = json.loads(raw)
            if doc_id in all_maps:
                mapping = all_maps[doc_id]
                self._chapter_map_cache[doc_id] = mapping
                return mapping
        chapters = self.list_chapters(doc_id)
        mapping = self._build_chapter_map(doc_id, chapters, self.make_source_uri)
        self._chapter_map_cache[doc_id] = mapping
        return mapping

    # ── Library / chapter browsing ──────────────────────────────────────────
