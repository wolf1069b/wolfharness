"""Entity read/write and symptom profile operations."""

from __future__ import annotations

from difflib import unified_diff
from hashlib import sha256
import json
import logging
import re
import time

from wolfharness.capabilities.wiki.auto_repair import materialize_body_links
from wolfharness.capabilities.wiki.io.text_parsers import (
    _ENTITY_URI_HEADING_RE,
)
from wolfharness.capabilities.wiki.quality import (
    entity_status,
    extract_sections,
    extract_source_uris,
    parse_frontmatter,
)
from wolfharness.capabilities.wiki.validation import (
    require_valid_entity,
)


logger = logging.getLogger(__name__)

from wolfharness.capabilities.wiki._helpers import (
    _FORMAL_WRITE_HOOKS,
    _conflicting_facts,
    _entity_batch_limit,
    _internal_conflicting_facts,
)


class EntityWriteMixin:
    """Entity read/write and symptom profile operations."""

    def list_entities(self, concept: str, class_name: str = "") -> list[dict]:
        """List entities — three-level progressive browsing.

        Usage (matches external MCP docstring)::

            list_entities()                → all concepts with entity counts
            list_entities("Component")     → classes under Component
            list_entities("Component", "主泵") → entities under Component/主泵

        Level 0 (empty *concept*) returns ``[{concept, total}, ...]``.
        Level 1/2 returns ``[{object_name, title, class_name, concept, uri}, ...]``.
        """
        if not concept:
            # Level 0: list all concepts with entity counts
            counts: dict[str, int] = {}
            for c, _cls, _obj, _uri in self.store.list_entities(None):
                counts[c] = counts.get(c, 0) + 1
            return [
                {"concept": c, "total": n} for c, n in sorted(counts.items(), key=lambda x: -x[1])
            ]

        out: list[dict[str, str]] = []
        for c, cls, obj, uri in self.store.list_entities(concept):
            logical_class = self.store.logical_class_name(c, cls) or ""
            if class_name and logical_class != class_name:
                continue
            out.append(
                {
                    "object_name": obj,
                    "title": obj,
                    "class_name": logical_class,
                    "concept": c,
                    "uri": uri,
                },
            )
        return out

    def read_entity(self, concept: str, class_name: str, object_name: str) -> str:
        """Read the full content of an existing entity; raises if absent."""
        content = self.store.read_entity(concept, class_name or None, object_name)
        if content is None:
            raise FileNotFoundError(
                f"Entity not found: {self.store.entity_uri(concept, class_name or None, object_name)}",
            )
        return content

    def _mark_merge_conflict(
        self,
        concept: str,
        uri: str,
        current: str,
        candidate: str,
    ) -> str:
        """Record a fact-changing merge and keep the candidate unpublished."""
        if concept not in {"Fault", "Symptom", "Component"}:
            return candidate
        changed_facts = sorted(
            _conflicting_facts(current, candidate) | _internal_conflicting_facts(candidate)
        )
        if not changed_facts:
            return candidate
        evidence = list(
            dict.fromkeys(
                [
                    *extract_source_uris(current),
                    *extract_source_uris(candidate),
                    uri,
                ],
            ),
        )
        entity_name = uri.rstrip("/").rsplit("/", 1)[-1].removesuffix(".md") or concept
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", entity_name).strip("-") or "entity"
        conflict_key = uri + chr(31).join(changed_facts)
        conflict_digest = sha256(conflict_key.encode()).hexdigest()[:12]
        opa_id = f"opa-merge-{slug}-{conflict_digest}"
        record = self.create_opa(
            opa_id=opa_id,
            title=f"{entity_name} — {concept} 增量合并事实冲突",
            description=f"同一实体的重复构建输入产生了不同事实：{uri}",
            category="conflict",
            reason_code="fact_conflict",
            target_uri=uri,
            target_section="增量合并事实",
            evidence_uris=evidence,
            finding="以下内容在重复构建中出现了不一致：\n" + "\n".join(changed_facts[:20]),
            missing="在人工裁决前，无法确认哪一组事实适用于当前机型或配置。",
            recommendation="保留双方证据，按机型和来源裁决；不得静默覆盖旧事实。",
        )
        marked = self._set_frontmatter_value(
            self._set_frontmatter_value(candidate, "conflict_pending", "true"),
            "conflict_refs",
            json.dumps([record["uri"]], ensure_ascii=False),
        )
        marker = f"- 待裁决冲突详情：{record['uri']}"
        if marker not in marked:
            suffix = "\n" if marked.endswith("\n") else "\n\n"
            marked = f"{marked}{suffix}## 冲突说明\n\n{marker}\n"
        return marked

    @staticmethod
    def _set_frontmatter_value(content: str, key: str, value: str) -> str:
        """Set one scalar/list-compatible frontmatter field deterministically."""
        lines = content.splitlines(keepends=True)
        if not lines or lines[0].strip() != "---":
            return content
        end = next(
            (index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"), None
        )
        if end is None:
            return content
        rendered = f"{key}: {value}\n"
        for index in range(1, end):
            if re.match(rf"^{re.escape(key)}\s*:", lines[index]):
                lines[index] = rendered
                return "".join(lines)
        lines.insert(end, rendered)
        return "".join(lines)

    def _assert_expert_authority_preserved(
        self,
        *,
        target_uri: str,
        current: str,
        candidate: str,
        reference_replacements: list[tuple[str, str]] | None = None,
    ) -> None:
        """Verify that expert-confirmed sections are not removed by a candidate write."""
        try:
            authorities = self.get_expert_authority(target_uri=target_uri)
        except (OSError, ValueError, KeyError):
            return
        if not authorities:
            return
        current_sections = {name for name, _ in extract_sections(current)}
        for authority in authorities:
            section = authority.get("target_section", "")
            if section and section in current_sections:
                candidate_sections = {name for name, _ in extract_sections(candidate)}
                if section not in candidate_sections:
                    raise ValueError(
                        f"Expert-confirmed section '{section}' would be removed from {target_uri}",
                    )

    def write_entity(
        self,
        concept: str,
        class_name: str,
        object_name: str,
        content: str,
        *,
        expected_sha256: str = "",
        conflict_policy: str = "detect",
        skip_materialization: bool = False,
        reference_replacements: list[tuple[str, str]] | None = None,
    ) -> str:
        """Write an entity file with body-reference resolution.

        A missing entity may be created directly.  An existing entity must be
        written through the diff/merge protocol and therefore requires the
        SHA-256 observed by the caller.  This prevents an extraction or
        relation worker from silently overwriting a newer page revision.

        Returns the canonical ``{root_uri}/...`` URI (hash-based for non-ASCII
        object names).
        """
        if conflict_policy not in {"detect", "external_authority"}:
            raise ValueError(f"Unsupported conflict_policy: {conflict_policy!r}")
        self._invalidate_audit_cache()
        clz = class_name or None
        uri = self.store.entity_uri(concept, clz, object_name)
        current = self.store.read_entity(concept, clz, object_name)
        if current is not None:
            current_sha256 = sha256(current.encode("utf-8")).hexdigest()
            if not expected_sha256:
                raise ValueError(
                    "Existing entity writes require diff_entity followed by merge_entity(expected_sha256=...)",
                )
            if expected_sha256 != current_sha256:
                raise ValueError(
                    f"Entity changed since diff; rerun diff_entity before merge (expected={expected_sha256}, actual={current_sha256}).",
                )
        # Keep the entity's own identity before dropping unresolved references.
        # Extraction may emit a stale path-shaped URI in this heading.
        content = _ENTITY_URI_HEADING_RE.sub(f"# {uri}\n", content, count=1)
        # Repair YAML frontmatter (concatenation, [简述] prefix, dedup, hash URIs).
        content = self.store.repair_frontmatter(
            content,
            None,
        )
        # Resolve human-readable viking:// URIs in body content to hash-based URIs.
        content = self.store.resolve_body_refs(
            content,
            None,
        )
        materialization_started = time.perf_counter()
        # Materialize relation-field URIs in their canonical body sections at
        # write time. This makes new incremental pages self-contained and
        # prevents a later audit from treating an already-known relation as a
        # dangling body-link gap. Unknown targets are left untouched by the
        # repair helper and remain source-honest open gaps.
        content = materialize_body_links(content, concept, self)
        self._reject_wrong_raw_refs(content)
        self._reject_malformed_wiki_refs(content)
        self._reject_nonexistent_raw_sources(content)
        # Deduplicate inline [viking://uri] citations within each section.
        content = self.store.dedup_citations(content)
        content = self._dedupe_h2_sections(content)
        if current is not None and conflict_policy == "detect":
            self._assert_expert_authority_preserved(
                target_uri=uri,
                current=current,
                candidate=content,
                reference_replacements=reference_replacements,
            )
        self._validate_formal_write(
            content=content,
            concept=concept,
            class_name=class_name,
            object_name=object_name,
            skip_materialization=skip_materialization,
        )
        self._record_phase_timing("materialization", materialization_started)
        existing_content = self.store.read_entity(concept, clz, object_name)
        if existing_content is not None and conflict_policy == "detect":
            content = self._mark_merge_conflict(concept, uri, existing_content, content)
        elif existing_content is not None and conflict_policy == "external_authority":
            content = self._set_frontmatter_value(content, "conflict_pending", "false")
            content = self._set_frontmatter_value(content, "conflict_refs", "[]")
        is_new = existing_content is None
        char_before = len(existing_content) if existing_content is not None else 0
        parent_existed = self.store.is_dir(
            self.store._key_of(self.store.entity_path(concept, clz, object_name).parent),
        )
        if self._log:
            self._log.mutation_attempt(uri, "write_entity")
        # Recheck under the same process-wide store lock used by expert
        # authority claims.  The initial optimistic check may have happened
        # before an expert OPL was reserved/applied by another worker.
        with self._relation_sync_lock:
            latest = self.store.read_entity(concept, clz, object_name)
            if current is None and latest is not None:
                raise ValueError(
                    "Entity was created concurrently; rerun diff_entity before writing"
                )
            if current is not None:
                latest_sha256 = sha256((latest or "").encode("utf-8")).hexdigest()
                if latest_sha256 != expected_sha256:
                    raise ValueError(
                        f"Entity changed before commit; rerun diff_entity before merge (expected={expected_sha256}, actual={latest_sha256}).",
                    )
                if conflict_policy == "detect":
                    self._assert_expert_authority_preserved(
                        target_uri=uri,
                        current=latest or "",
                        candidate=content,
                        reference_replacements=reference_replacements,
                    )
            self.store.write_entity(concept, clz, object_name, content)
        if self._log:
            self._log.mutation_applied(uri, "write_entity")
        # Register natural-key with class_name for collision-free lookups.
        self.store.register_natural_key(concept, clz, object_name, uri)
        if self._log:
            if is_new:
                self._log.entity_created(
                    uri,
                    concept,
                    object_name,
                    len(content),
                    class_name=clz,
                    folder_created=not parent_existed,
                )
            else:
                self._log.entity_updated(
                    uri,
                    concept,
                    object_name,
                    len(content),
                    char_before,
                    class_name=clz,
                    reason="merge",
                )
        logger.info("Entity written: %s (%d chars)", uri, len(content))
        relation_started = time.perf_counter()
        if concept == "Component":
            self._sync_component_narrative_links(uri)
        elif concept in {"Fault", "Procedure"}:
            relation_field = "affected_components" if concept == "Fault" else "target_components"
            title = parse_frontmatter(content).get("title")
            label = title.strip() if isinstance(title, str) and title.strip() else object_name
            for component_uri in self._frontmatter_uri_values(
                self.store.root_uri, parse_frontmatter(content), relation_field
            ):
                canonical_component_uri = component_uri.split("#", 1)[0]
                if concept == "Fault":
                    self._sync_component_narrative_links(
                        canonical_component_uri,
                        fault_links=[(label, uri)],
                    )
                else:
                    self._sync_component_narrative_links(
                        canonical_component_uri,
                        procedure_links=[(label, uri)],
                    )
        self._record_phase_timing("relation", relation_started)
        return uri

    def write_entities_batch(self, entities: list[dict[str, object]]) -> dict[str, object]:
        """Validate and materialize a bounded set of independent entities.

        The batch removes one model round trip per entity. Each item keeps the
        same optimistic-lock contract as ``write_entity`` through an optional
        ``expected_sha256`` field. Component narrative mirrors are deferred
        until every entity body is visible, then synchronized once per target
        component.
        """
        if not entities:
            raise ValueError("entities must not be empty")
        batch_limit = _entity_batch_limit()
        if len(entities) > batch_limit:
            raise ValueError(f"write_entities_batch accepts at most {batch_limit} entities")

        materialization_started = time.perf_counter()
        prepared: list[tuple[str, str, str, str, str, int | None]] = []
        identities: set[tuple[str, str, str]] = set()
        for index, item in enumerate(entities):
            concept = item.get("concept")
            class_name = item.get("class_name", "")
            object_name = item.get("object_name")
            content = item.get("content")
            expected_sha256 = item.get("expected_sha256", "")
            if not all(
                isinstance(value, str)
                for value in (concept, class_name, object_name, content, expected_sha256)
            ):
                raise TypeError(f"entities[{index}] fields must be strings")
            assert isinstance(concept, str)
            assert isinstance(class_name, str)
            assert isinstance(object_name, str)
            assert isinstance(content, str)
            assert isinstance(expected_sha256, str)
            identity = (concept, class_name, object_name)
            if identity in identities:
                raise ValueError(f"duplicate entity identity in batch: {identity}")
            identities.add(identity)

            current = self.store.read_entity(concept, class_name or None, object_name)
            if current is not None:
                current_hash = sha256(current.encode("utf-8")).hexdigest()
                if not expected_sha256:
                    raise ValueError(
                        f"Existing batch entity requires expected_sha256: {identity}",
                    )
                if expected_sha256 != current_hash:
                    raise ValueError(f"Batch entity changed since diff: {identity}")
            normalized = _ENTITY_URI_HEADING_RE.sub(
                f"# {self.store.entity_uri(concept, class_name or None, object_name)}\n",
                content,
                count=1,
            )
            normalized = self.store.repair_frontmatter(normalized, None)
            normalized = self.store.resolve_body_refs(normalized, None)
            self._reject_wrong_raw_refs(normalized)
            self._reject_malformed_wiki_refs(normalized)
            normalized = self.store.dedup_citations(normalized)
            normalized = self._dedupe_h2_sections(normalized)
            self._validate_formal_write(
                content=normalized,
                concept=concept,
                class_name=class_name,
                object_name=object_name,
            )
            current_content = self.store.read_entity(concept, class_name or None, object_name)
            if current_content is not None:
                normalized = self._mark_merge_conflict(
                    concept,
                    self.store.entity_uri(concept, class_name or None, object_name),
                    current_content,
                    normalized,
                )
            prepared.append(
                (
                    concept,
                    class_name,
                    object_name,
                    normalized,
                    expected_sha256,
                    len(current_content) if current_content is not None else None,
                ),
            )

        # All identities in the batch are already known before publication.
        # Let deterministic body-link materialization use that set so links
        # between two new entities do not require a write-read-write cycle.
        batch_uris = {
            self.store.entity_uri(concept, class_name or None, object_name)
            for concept, class_name, object_name, _content, _expected_sha256, _char_before in prepared
        }
        prepared = [
            (
                concept,
                class_name,
                object_name,
                materialize_body_links(content, concept, self, known_uris=batch_uris),
                expected_sha256,
                char_before,
            )
            for concept, class_name, object_name, content, expected_sha256, char_before in prepared
        ]

        uris: list[str] = []
        sync_targets: set[str] = set()
        if self._log:
            for (
                concept,
                class_name,
                object_name,
                _content,
                _expected_sha256,
                _char_before,
            ) in prepared:
                self._log.mutation_attempt(
                    self.store.entity_uri(concept, class_name or None, object_name),
                    "write_entities_batch",
                )
        self.store.write_entities(
            [
                (concept, class_name or None, object_name, content)
                for concept, class_name, object_name, content, _expected_sha256, _char_before in prepared
            ],
        )
        for concept, class_name, object_name, content, _expected_sha256, char_before in prepared:
            uri = self.store.entity_uri(concept, class_name or None, object_name)
            self.store.register_natural_key(concept, class_name or None, object_name, uri)
            if self._log:
                self._log.mutation_applied(uri, "write_entities_batch")
            body_content = content
            uris.append(uri)
            if concept == "Component":
                sync_targets.add(uri)
            elif concept in {"Fault", "Procedure"}:
                relation_field = (
                    "affected_components" if concept == "Fault" else "target_components"
                )
                frontmatter = parse_frontmatter(body_content)
                sync_targets.update(
                    value.split("#", 1)[0]
                    for value in self._frontmatter_uri_values(
                        self.store.root_uri,
                        frontmatter,
                        relation_field,
                    )
                )
            if self._log:
                if char_before is None:
                    self._log.entity_created(
                        uri,
                        concept,
                        object_name,
                        len(body_content),
                        class_name=class_name or None,
                    )
                else:
                    self._log.entity_updated(
                        uri,
                        concept,
                        object_name,
                        len(body_content),
                        char_before,
                        class_name=class_name or None,
                        reason="batch_merge",
                    )
        self._record_phase_timing("materialization", materialization_started)
        relation_started = time.perf_counter()
        for component_uri in sorted(sync_targets):
            self._sync_component_narrative_links(component_uri)
        self._record_phase_timing("relation", relation_started)
        self._invalidate_audit_cache()
        return {
            "written_count": len(uris),
            "skipped_count": 0,
            "uris": uris,
        }

    def write_symptom_profile(
        self,
        symptom_uri: str,
        profile_id: str,
        content: str,
        *,
        expected_sha256: str = "",
        skip_materialization: bool = False,
    ) -> str:
        """Create a Profile or guarded-merge an existing Profile."""
        profile_uri = self.store.symptom_profile_uri(symptom_uri, profile_id)
        current = self.store.read_entity_by_uri(profile_uri)
        if current is not None:
            current_sha256 = sha256(current.encode("utf-8")).hexdigest()
            if not expected_sha256:
                raise ValueError(
                    "Existing Symptom Profile writes require diff_entity followed by a guarded merge",
                )
            if expected_sha256 != current_sha256:
                raise ValueError(
                    f"Symptom Profile changed since diff; reread before merge (expected={expected_sha256}, actual={current_sha256}).",
                )
        content = self.store.repair_frontmatter(content, None)
        content = self.store.resolve_body_refs(content, None)
        self._reject_wrong_raw_refs(content)
        self._reject_malformed_wiki_refs(content)
        content = self.store.dedup_citations(content)
        content = self._dedupe_h2_sections(content)
        parent_info = self.store.lookup_by_uri(symptom_uri)
        if parent_info is None or parent_info[0] != "Symptom":
            raise ValueError(f"Unknown canonical Symptom URI: {symptom_uri}")
        if entity_status(content) == "confirmed":
            require_valid_entity(
                content=content,
                concept="Symptom",
                class_name=parent_info[1] or "",
                object_name=parent_info[2],
                hooks=_FORMAL_WRITE_HOOKS,
            )
        else:
            self._validate_formal_write(
                content=content,
                concept="Symptom",
                class_name=parent_info[1] or "",
                object_name=parent_info[2],
                skip_materialization=skip_materialization,
            )
        self.store.write_symptom_profile(symptom_uri, profile_id, content)
        # Backfill parent index.md with updated Profile 索引
        self._sync_symptom_profile_index(symptom_uri)
        logger.info("Symptom Profile written: %s (%d chars)", profile_uri, len(content))
        return profile_uri

    def patch_symptom_profile(
        self,
        symptom_uri: str,
        profile_id: str,
        operations: list[dict],
        *,
        expected_sha256: str = "",
    ) -> str:
        """Patch one formal Symptom Profile with optimistic locking."""
        profile_uri = self.store.symptom_profile_uri(symptom_uri, profile_id)
        content = self.store.read_entity_by_uri(profile_uri)
        if content is None:
            raise FileNotFoundError(f"Symptom Profile not found: {profile_uri}")
        current_sha256 = sha256(content.encode("utf-8")).hexdigest()
        if not expected_sha256:
            raise ValueError(
                "Existing Symptom Profile patches require expected_sha256",
            )
        if expected_sha256 != current_sha256:
            raise ValueError(
                f"Symptom Profile changed before patch; reread before patch (expected={expected_sha256}, actual={current_sha256}).",
            )
        content = self._apply_operations(content, operations)
        content = self.store.repair_frontmatter(content, None)
        content = self.store.resolve_body_refs(content, None)
        self._reject_wrong_raw_refs(content)
        self._reject_malformed_wiki_refs(content)
        content = self.store.dedup_citations(content)
        content = self._dedupe_h2_sections(content)
        if entity_status(content) == "confirmed":
            parent_info = self.store.lookup_by_uri(symptom_uri)
            if parent_info is None:
                raise ValueError(f"Unknown parent Symptom: {symptom_uri}")
            _concept, class_name, object_name = parent_info
            require_valid_entity(
                content=content,
                concept="Symptom",
                class_name=class_name or "",
                object_name=object_name,
                hooks=_FORMAL_WRITE_HOOKS,
            )
        else:
            parent_info = self.store.lookup_by_uri(symptom_uri)
            if parent_info is None:
                raise ValueError(f"Unknown parent Symptom: {symptom_uri}")
            self._validate_formal_write(
                content=content,
                concept="Symptom",
                class_name=parent_info[1] or "",
                object_name=parent_info[2],
            )
        if content == self.store.read_entity_by_uri(profile_uri):
            return profile_uri
        self.write_symptom_profile(
            symptom_uri,
            profile_id,
            content,
            expected_sha256=current_sha256,
        )
        return profile_uri

    def list_symptom_profiles(self, symptom_uri: str) -> list[dict[str, str]]:
        """List Profile subresources belonging to a canonical Symptom."""
        return [
            {"profile_id": profile_id, "uri": uri}
            for profile_id, uri in self.store.list_symptom_profiles(symptom_uri)
        ]

    def diff_symptom_profile(
        self,
        symptom_uri: str,
        profile_id: str,
        candidate_content: str,
    ) -> dict[str, object]:
        """Return a deterministic diff and revision hashes for a Profile."""
        profile_uri = self.store.symptom_profile_uri(symptom_uri, profile_id)
        current = self.store.read_entity_by_uri(profile_uri)
        current_hash = sha256((current or "").encode("utf-8")).hexdigest()
        candidate_hash = sha256(candidate_content.encode("utf-8")).hexdigest()
        diff = "".join(
            unified_diff(
                (current or "").splitlines(keepends=True),
                candidate_content.splitlines(keepends=True),
                fromfile="current",
                tofile="candidate",
            ),
        )
        return {
            "uri": profile_uri,
            "exists": current is not None,
            "changed": current != candidate_content,
            "current_sha256": current_hash,
            "candidate_sha256": candidate_hash,
            "diff": diff,
        }

    def migrate_legacy_symptom_profiles(self) -> dict[str, int]:
        """Move legacy pseudo-Symptom Profile entities into ``profile/``."""
        symptom_root_key = self.store.CONCEPT_DIRS["Symptom"]
        migrated = 0
        skipped = 0
        for key in self.store.list_dir(symptom_root_key, recursive=True):
            if not key.endswith("/index.md"):
                continue
            content = self.store.read_text(key)
            if content is None:
                continue
            profile_match = re.search(r"^profile_id:\s*([^\s]+)\s*$", content, re.MULTILINE)
            parent_match = re.search(
                rf"^parent_symptom:\s*({re.escape(self.store.root_uri)}/Symptom/\S+)\s*$",
                content,
                re.MULTILINE,
            )
            if profile_match is None or parent_match is None:
                continue
            profile_id = profile_match.group(1)
            symptom_uri = parent_match.group(1)
            try:
                target_path = self.store.symptom_profile_path(symptom_uri, profile_id)
            except ValueError:
                skipped += 1
                continue
            target_key = self.store._key_of(target_path)
            target_content = self.store.read_text(target_key)
            if target_content is not None and target_content != content:
                skipped += 1
                continue

            legacy_uri: str | None = None
            for concept, class_name, object_name, uri in self.store.list_entities("Symptom"):
                if (
                    self.store._key_of(self.store.entity_path(concept, class_name, object_name))
                    == key
                ):
                    legacy_uri = uri
                    break

            target_content = self.store.read_entity_by_uri(
                self.store.symptom_profile_uri(symptom_uri, profile_id),
            )
            expected_sha256 = (
                sha256(target_content.encode("utf-8")).hexdigest()
                if target_content is not None
                else ""
            )
            self.write_symptom_profile(
                symptom_uri,
                profile_id,
                content,
                expected_sha256=expected_sha256,
            )
            if key != target_key:
                self.store.delete(key)
                # ponytail: no empty-dir cleanup on backend keys; Viking has no dirs.
            if legacy_uri is not None and legacy_uri != symptom_uri:
                self.store.unregister_uri(legacy_uri)
            migrated += 1
        return {"migrated": migrated, "skipped": skipped}

    # ── Patch-based entity editing ─────────────────────────────────────────
