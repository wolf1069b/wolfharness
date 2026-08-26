"""BOM taxonomy and component registration."""

from __future__ import annotations

from hashlib import sha256
import json
import logging
import re

from httpx import HTTPError
from openviking_sdk.errors import OpenVikingError

from wolfharness.capabilities.wiki.section_constants import (
    SECTION_MECHANISM,
    SECTION_OVERVIEW,
    SECTION_SOURCE,
)


logger = logging.getLogger(__name__)

from wolfharness.capabilities.wiki._helpers import _BOM_PATH_PLACEHOLDER_RE, _entity_batch_limit


class BomMixin:
    """BOM taxonomy and component registration."""

    def _bom_taxonomy_key(self) -> str:
        """Return the persistent global BOM taxonomy registry key."""
        return "index/bom_taxonomy.json"

    def get_bom_taxonomy(self) -> dict[str, dict[str, object]]:
        """Read the global BOM Component registry without changing it."""
        raw = self.store.read_text(self._bom_taxonomy_key())
        if raw is None:
            return {}
        try:
            value = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid BOM taxonomy registry: {self._bom_taxonomy_key()}") from exc
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise TypeError(f"Invalid BOM taxonomy registry: {self._bom_taxonomy_key()}")
        return {str(uri): record for uri, record in value.items() if isinstance(record, dict)}

    def register_bom_component(
        self,
        component_uri: str,
        bom_path: str,
        *,
        parent_uri: str = "",
        evidence_uris: list[str] | None = None,
        applicable_models: list[str] | None = None,
    ) -> dict[str, object]:
        """Register one global BOM Component identity and its evidence-backed path."""
        identity = self.store.lookup_by_uri(component_uri)
        if identity is None or identity[0] != "Component":
            raise ValueError(f"Unknown Component URI: {component_uri}")
        normalized_path = self.store.normalize_class_name("Component", bom_path.strip()) or ""
        if not normalized_path:
            raise ValueError("bom_path must not be empty")
        if normalized_path.startswith(("关重件/", "普通件/")):
            raise ValueError("bom_path must be a logical BOM path, not a physical tier")
        evidence = list(dict.fromkeys(uri.strip() for uri in (evidence_uris or []) if uri.strip()))
        raw_prefix = self._raw_fs.root_uri + "/"
        bom_prefix = (self._bom_fs.root_uri + "/") if self._bom_fs is not None else ""
        if not evidence or any(
            not (uri.startswith(raw_prefix) or bool(bom_prefix and uri.startswith(bom_prefix)))
            for uri in evidence
        ):
            raise ValueError(
                f"BOM registration requires evidence rooted at the raw ({raw_prefix}) or global BOM ({bom_prefix or '<!-- unset -->'}) namespace",
            )
        unresolved_evidence = [uri for uri in evidence if self.read_resource(uri) is None]
        if unresolved_evidence:
            raise ValueError(f"BOM evidence cannot be resolved: {unresolved_evidence[:5]}")
        if parent_uri:
            parent_identity = self.store.lookup_by_uri(parent_uri)
            if parent_identity is None or parent_identity[0] != "Component":
                raise ValueError(f"Unknown BOM parent Component URI: {parent_uri}")
            if parent_uri == component_uri:
                raise ValueError("A Component cannot be its own BOM parent")
        registry = self.get_bom_taxonomy()
        existing = registry.get(component_uri)
        if existing is not None and str(existing.get("bom_path", "")) != normalized_path:
            raise ValueError(
                f"BOM path conflict for {component_uri}: {existing.get('bom_path', '')!r} vs {normalized_path!r}; create an OPA first",
            )
        record: dict[str, object] = {
            "component_uri": component_uri,
            "bom_path": normalized_path,
            "parent_uri": parent_uri,
            "evidence_uris": evidence,
            "applicable_models": list(dict.fromkeys(applicable_models or [])),
        }
        registry[component_uri] = record
        self.store.write_text(
            self._bom_taxonomy_key(),
            json.dumps(dict(sorted(registry.items())), ensure_ascii=False, indent=2) + "\n",
        )
        return record

    def register_bom_identity_batch(
        self,
        doc_id: str,
        packet_id: str,
        bom_source_uri: str,
        target_model: str,
        components: list[dict[str, object]],
        *,
        device_object_name: str = "",
        device_description: str = "",
        system_chapter_uris: list[str] | None = None,
    ) -> dict[str, object]:
        """Persist a complete BOM identity plan in one idempotent operation.

        A BOM is control metadata, not a chapter-extraction workload.  This
        operation deliberately keeps BOM handling on the service side: it
        records one immutable packet, materializes deterministic Component and
        Device skeletons in bounded storage batches, and commits the taxonomy
        registry once.  It must not be decomposed into distill or worker tasks.

        ``components`` is the already-resolved identity list from the exact
        BOM leaf: each item requires ``class_name`` and ``object_name`` and may
        provide ``bom_path``, ``parent_uri``, ``evidence_uris`` and
        ``applicable_models``. Semantic mechanism enrichment is a separate
        guarded worker patch after this identity commit.
        """
        if not doc_id.strip() or not packet_id.strip() or not target_model.strip():
            raise ValueError("doc_id, packet_id, and target_model must not be empty")
        if not re.fullmatch(r"[A-Za-z0-9_]+", packet_id):
            raise ValueError("packet_id must contain only ASCII letters, digits, and underscores")
        bom_uri = bom_source_uri.strip()
        if not bom_uri or self.read_resource(bom_uri) is None:
            raise ValueError(f"BOM source is unreadable: {bom_uri!r}")
        if self._bom_fs is not None and not (
            bom_uri == self._bom_fs.root_uri
            or bom_uri.startswith(self._bom_fs.root_uri.rstrip("/") + "/")
        ):
            raise ValueError("bom_source_uri must belong to the configured BOM namespace")
        if len(components) > 1000:
            raise ValueError("BOM identity batch accepts at most 1000 components")

        normalized_components: list[dict[str, object]] = []
        uri_by_identity: dict[tuple[str, str], str] = {}
        for index, item in enumerate(components):
            if not isinstance(item, dict):
                raise TypeError(f"components[{index}] must be an object")
            class_name = item.get("class_name")
            object_name = item.get("object_name")
            if not isinstance(class_name, str) or not class_name.strip():
                raise ValueError(f"components[{index}].class_name must not be empty")
            if not isinstance(object_name, str) or not object_name.strip():
                raise ValueError(f"components[{index}].object_name must not be empty")
            normalized_class = self.store.normalize_class_name("Component", class_name.strip())
            normalized_object = object_name.strip()
            if _BOM_PATH_PLACEHOLDER_RE.search(normalized_class):
                raise ValueError(
                    f"components[{index}].class_name contains a placeholder or ellipsis",
                )
            identity_key = (normalized_class, normalized_object)
            if identity_key in uri_by_identity:
                raise ValueError(f"Duplicate BOM Component identity: {identity_key}")
            component_uri = self.store.entity_uri("Component", normalized_class, normalized_object)
            uri_by_identity[identity_key] = component_uri
            raw_evidence = item.get("evidence_uris", [bom_uri])
            if not isinstance(raw_evidence, list) or not all(
                isinstance(value, str) for value in raw_evidence
            ):
                raise TypeError(f"components[{index}].evidence_uris must be a string list")
            evidence = list(
                dict.fromkeys([
                    bom_uri,
                    *(value.strip() for value in raw_evidence if value.strip()),
                ])
            )
            raw_models = item.get("applicable_models", [target_model])
            if not isinstance(raw_models, list) or not all(
                isinstance(value, str) for value in raw_models
            ):
                raise TypeError(f"components[{index}].applicable_models must be a string list")
            models = list(
                dict.fromkeys([value.strip() for value in raw_models if value.strip()])
            ) or [target_model.strip()]
            bom_path = item.get("bom_path", normalized_class)
            if not isinstance(bom_path, str) or not bom_path.strip():
                raise ValueError(f"components[{index}].bom_path must not be empty")
            normalized_path = self.store.normalize_class_name("Component", bom_path.strip())
            if _BOM_PATH_PLACEHOLDER_RE.search(normalized_path):
                raise ValueError(
                    f"components[{index}].bom_path contains a placeholder or ellipsis",
                )
            if normalized_path != normalized_class:
                raise ValueError(
                    f"components[{index}].class_name must equal its logical bom_path",
                )
            parent_uri = item.get("parent_uri", "")
            if not isinstance(parent_uri, str):
                raise TypeError(f"components[{index}].parent_uri must be a string")
            normalized_components.append(
                {
                    "component_uri": component_uri,
                    "class_name": normalized_class,
                    "object_name": normalized_object,
                    "bom_path": normalized_path,
                    "parent_uri": parent_uri.strip(),
                    "evidence_uris": evidence,
                    "applicable_models": models,
                },
            )

        component_errors: list[dict[str, str]] = []
        eligible_components: list[dict[str, object]] = []
        batch_component_uris = {str(value["component_uri"]) for value in normalized_components}
        # Resolve optional parent identities without another model/tool round
        # trip.  A parent may be an exact URI or an identity in this batch.
        for item in normalized_components:
            parent_uri = str(item["parent_uri"])
            if (
                parent_uri
                and parent_uri not in batch_component_uris
                and self.store.lookup_by_uri(parent_uri) is None
            ):
                component_errors.append({
                    "component_uri": str(item["component_uri"]),
                    "error": f"Unknown BOM parent Component URI: {parent_uri}",
                })
                continue
            if parent_uri == item["component_uri"]:
                component_errors.append({
                    "component_uri": str(item["component_uri"]),
                    "error": "A Component cannot be its own BOM parent",
                })
                continue
            eligible_components.append(item)

        registry = self.get_bom_taxonomy()
        validated_components: list[dict[str, object]] = []
        eligible_uris = {str(value["component_uri"]) for value in eligible_components}
        readable_evidence: dict[str, bool] = {bom_uri: True}
        for item in eligible_components:
            component_uri = str(item["component_uri"])
            parent_uri = str(item["parent_uri"])
            if (
                parent_uri
                and parent_uri in batch_component_uris
                and parent_uri not in eligible_uris
            ):
                component_errors.append({
                    "component_uri": component_uri,
                    "error": f"BOM parent is invalid in this batch: {parent_uri}",
                })
                continue
            evidence = list(item["evidence_uris"])
            if not evidence:
                component_errors.append({
                    "component_uri": component_uri,
                    "error": "BOM evidence is empty",
                })
                continue
            evidence_error = ""
            for evidence_uri in evidence:
                evidence_readable = readable_evidence.get(evidence_uri)
                if evidence_readable is None:
                    evidence_readable = self.read_resource(evidence_uri) is not None
                    readable_evidence[evidence_uri] = evidence_readable
                if not evidence_readable:
                    evidence_error = f"BOM evidence cannot be resolved: {evidence_uri}"
                    break
            if evidence_error:
                component_errors.append({"component_uri": component_uri, "error": evidence_error})
                continue
            existing = registry.get(component_uri)
            if existing is not None and str(existing.get("bom_path", "")) != str(item["bom_path"]):
                component_errors.append(
                    {
                        "component_uri": component_uri,
                        "error": f"BOM path conflict: {existing.get('bom_path', '')!r} vs {item['bom_path']!r}; create an OPA first",
                    },
                )
                continue
            validated_components.append(item)

        # Persist the source-of-truth packet before any entity mutation.  A
        # changed packet_id/source hash is rejected by record_source_packet,
        # making retries safe and preventing duplicate identity plans.  All
        # evidence and registry conflicts are validated above so a rejected
        # batch cannot leave a misleading packet behind.
        packet = self.record_source_packet(
            packet_id=packet_id,
            doc_id=doc_id,
            source_uris=[bom_uri],
            packet_body={
                "kind": "bom_identity_plan",
                "target_model": target_model.strip(),
                "resolved_components": normalized_components,
                "resolved_system_chapter_uris": list(dict.fromkeys(system_chapter_uris or [])),
            },
            allow_same_snapshot_replace=True,
        )

        entity_items: list[dict[str, object]] = []
        for item in validated_components:
            component_uri = str(item["component_uri"])
            if self.store.read_entity_by_uri(component_uri) is not None:
                continue
            object_name = str(item["object_name"])
            class_name = str(item["class_name"])
            bom_path = str(item["bom_path"])
            evidence = [str(value) for value in item["evidence_uris"]]
            models = [str(value) for value in item["applicable_models"]]
            parent_uri = str(item["parent_uri"])
            parent_field = f"bom_parent_uri: {parent_uri}\n" if parent_uri else ""
            evidence_yaml = "\n".join(f"  - {value}" for value in evidence)
            models_yaml = "\n".join(f"  - {value}" for value in models)
            entity_items.append(
                {
                    "concept": "Component",
                    "class_name": class_name,
                    "object_name": object_name,
                    "content": (
                        "---\n"
                        f"id: BOMCOMP-{sha256(component_uri.encode('utf-8')).hexdigest()[:16]}\n"
                        f"title: {object_name}\n"
                        f"description: {target_model.strip()} BOM 中确认的 {bom_path} 总成身份。\n"
                        f"class_name: {class_name}\n"
                        f"object_name: {object_name}\n"
                        f"bom_path: {bom_path}\n"
                        f"{parent_field}"
                        "bom_evidence:\n"
                        f"{evidence_yaml}\n"
                        "status: draft\n"
                        "applicable_models:\n"
                        f"{models_yaml}\n"
                        "---\n"
                        f"# {object_name}\n\n"
                        f"## {SECTION_OVERVIEW}\n"
                        f"BOM 已确认该总成属于 {bom_path}，适用机型：{'、'.join(models)}。\n\n"
                        f"## {SECTION_MECHANISM}\n"
                        "BOM 身份已登记；工作机理由 bom_enrich worker 在身份入库后基于总成类型和装配路径推理补充，当前阶段不写入未经确认的参数。\n\n"
                        f"## {SECTION_SOURCE}\n"
                        f"{evidence_yaml.replace('  - ', '- ')}\n"
                    ),
                },
            )

        device_name = device_object_name.strip() or target_model.strip()
        device_uri = self.store.entity_uri("Device", None, device_name)
        if self.store.read_entity_by_uri(device_uri) is None:
            component_uris = [str(item["component_uri"]) for item in validated_components]
            component_yaml = "\n".join(f"  - {uri}" for uri in component_uris)
            chapter_uris = list(
                dict.fromkeys(uri.strip() for uri in (system_chapter_uris or []) if uri.strip())
            )
            chapter_yaml = "\n".join(f"  - {uri}" for uri in chapter_uris)
            entity_items.append(
                {
                    "concept": "Device",
                    "class_name": "",
                    "object_name": device_name,
                    "content": (
                        "---\n"
                        f"id: DEVICE-{sha256(device_uri.encode('utf-8')).hexdigest()[:16]}\n"
                        f"title: {device_name}\n"
                        f"description: {device_description.strip() or f'{target_model.strip()} 设备型号基线。'}\n"
                        "status: draft\n"
                        "critical_components:\n"
                        f"{component_yaml}\n"
                        "sources:\n"
                        f"  - {bom_uri}\n"
                        + (f"system_chapters:\n{chapter_yaml}\n" if chapter_yaml else "")
                        + "applicable_models:\n"
                        f"  - {target_model.strip()}\n"
                        "---\n"
                        f"# {device_name}\n\n"
                        "## 基础信息\n"
                        f"{device_description.strip() or f'{target_model.strip()} 设备型号基线。'}\n\n"
                        "## 包含系统\n"
                        "BOM 组件树已登记，系统章节与诊断实体在后续章节抽取阶段补充。\n\n"
                        f"## {SECTION_SOURCE}\n"
                        f"- {bom_uri}\n"
                    ),
                },
            )

        written = 0
        for start in range(0, len(entity_items), _entity_batch_limit()):
            chunk = entity_items[start : start + _entity_batch_limit()]
            try:
                result = self.write_entities_batch(chunk)
            except (OSError, ValueError, TypeError, HTTPError, OpenVikingError) as exc:
                # Preserve per-component isolation without falling back to
                # worker fan-out: retry only the failed storage chunk as
                # single-item service operations and report the exact URI.
                for item in chunk:
                    try:
                        result = self.write_entities_batch([item])
                    except (OSError, ValueError, TypeError, HTTPError, OpenVikingError) as item_exc:
                        component_errors.append(
                            {
                                "component_uri": self.store.entity_uri(
                                    str(item["concept"]),
                                    str(item.get("class_name", "")) or None,
                                    str(item["object_name"]),
                                ),
                                "error": f"batch materialization failed after {type(exc).__name__}: {item_exc}",
                            },
                        )
                    else:
                        written += int(result.get("written_count", 0))
            else:
                written += int(result.get("written_count", 0))

        # Pull chapter references from the raw backend at Device creation
        # time so the Device page immediately carries its source chapter
        # inventory.  When the remote source (e.g. MCP) does not expose
        # chapter URIs, ``system_chapters`` is left empty and the status is
        # documented in the ``包含系统`` body section.
        try:
            device_chapter_sync = self.sync_device_system_chapters(doc_id, device_name)
        except (OSError, HTTPError, OpenVikingError) as exc:
            logger.warning("sync_device_system_chapters failed at BOM time: %s", exc)
            device_chapter_sync = {"status": "skipped", "reason": "sync_error", "chapter_count": 0}

        sync_status = str(device_chapter_sync.get("status", ""))
        sync_reason = str(device_chapter_sync.get("reason", ""))
        chapter_count = int(device_chapter_sync.get("chapter_count", 0))

        if sync_status == "skipped" and sync_reason == "no_local_chapters":
            current_device = self.store.read_entity("Device", None, device_name)
            if current_device is not None and "BOM 组件树已登记" in current_device:
                current_hash = sha256(current_device.encode("utf-8")).hexdigest()
                self.patch_entity(
                    "Device",
                    "",
                    device_name,
                    [
                        {
                            "op": "section_replace",
                            "heading": "包含系统",
                            "content": (
                                "BOM 组件树已登记。\n\n"
                                "远端源未提供章节 URI，`system_chapters` 字段留空。"
                            ),
                        }
                    ],
                    expected_sha256=current_hash,
                )
        elif sync_status == "updated" and chapter_count > 0:
            current_device = self.store.read_entity("Device", None, device_name)
            if current_device is not None and "BOM 组件树已登记" in current_device:
                current_hash = sha256(current_device.encode("utf-8")).hexdigest()
                self.patch_entity(
                    "Device",
                    "",
                    device_name,
                    [
                        {
                            "op": "section_replace",
                            "heading": "包含系统",
                            "content": (
                                f"BOM 组件树已登记。已从远端源同步 {chapter_count} 个"
                                "系统章节引用（见 frontmatter `system_chapters`）。"
                            ),
                        }
                    ],
                    expected_sha256=current_hash,
                )

        for item in validated_components:
            component_uri = str(item["component_uri"])
            registry[component_uri] = {
                "component_uri": component_uri,
                "bom_path": str(item["bom_path"]),
                "parent_uri": str(item["parent_uri"]),
                "evidence_uris": list(item["evidence_uris"]),
                "applicable_models": list(item["applicable_models"]),
            }
        self.store.write_text(
            self._bom_taxonomy_key(),
            json.dumps(dict(sorted(registry.items())), ensure_ascii=False, indent=2) + "\n",
        )
        enrichment = self.bom_enrichment_status(packet_id)
        return {
            "status": "bom_registered",
            "event": "bom_registered",
            "packet_uri": packet["packet_uri"],
            "device_uri": device_uri,
            "component_uris": [str(item["component_uri"]) for item in validated_components],
            "component_count": len(validated_components),
            "written_count": written,
            "registry_count": len(registry),
            "component_errors": component_errors,
            "enrichment_status": enrichment.get("status", "pending"),
            "enrichment_pending_count": len(enrichment.get("pending_uris", [])),
        }
