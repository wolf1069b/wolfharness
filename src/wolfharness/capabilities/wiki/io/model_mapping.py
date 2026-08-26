"""Auditable engineering/sales model mapping for Wiki builds."""

from __future__ import annotations

from datetime import UTC, datetime
import re

from wolfharness.capabilities.wiki.quality import (
    RawSourceKind,
    SourceReadStatus,
    classify_raw_source_uri,
)
from wolfharness.capabilities.wiki.wiki_build_deps import WikiBuildDeps


class ModelMappingMixin(WikiBuildDeps):
    """Persist model aliases without turning unsupported guesses into facts."""

    _MODEL_MAPPING_KEY = "index/model_mapping.json"

    @staticmethod
    def _model_label(value: str) -> str:
        """Normalize a model label for matching while preserving display text."""
        return re.sub(r"[^A-Za-z0-9]+", "", value).casefold()

    def _read_model_mapping_records(self) -> list[dict[str, object]]:
        raw = self.store.read_json(self._MODEL_MAPPING_KEY)
        if raw is None:
            return []
        records = raw.get("mappings")
        if not isinstance(records, list):
            raise TypeError("Invalid model mapping registry: mappings must be a list")
        return [record for record in records if isinstance(record, dict)]

    @staticmethod
    def _unique_labels(values: list[str] | None) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in (values or []) if value.strip()))

    def _validate_mapping_sources(self, source_uris: list[str]) -> list[str]:
        valid: list[str] = []
        for uri in self._unique_labels(source_uris):
            kind = classify_raw_source_uri(uri, raw_root_uri=self._raw_fs.root_uri)
            if kind is None or kind is RawSourceKind.EXTERNAL:
                if self.read_resource(uri) is None:
                    raise ValueError(f"Model mapping evidence cannot be resolved: {uri}")
            else:
                result = self.read_raw_source(uri)
                if result.status is not SourceReadStatus.OK:
                    raise ValueError(f"Model mapping evidence cannot be resolved: {uri}")
            valid.append(uri)
        return valid

    def _find_device_uri(self, labels: list[str]) -> str:
        normalized = {self._model_label(label) for label in labels if label.strip()}
        for _concept, class_name, object_name, uri in self.store.list_entities("Device"):
            candidates = {self._model_label(object_name)}
            if class_name:
                candidates.add(self._model_label(class_name))
            if normalized.intersection(candidates):
                return uri
        return ""

    def register_model_mapping(
        self,
        engineering_model: str,
        sales_models: list[str],
        *,
        parent_model: str = "",
        device_uri: str = "",
        source_uris: list[str] | None = None,
        provenance: str = "manual",
        approved_by: str = "",
        notes: str = "",
    ) -> dict[str, object]:
        """Register one engineering/sales/series mapping.

        A mapping without readable source evidence is persisted as ``pending``.
        It can guide candidate lookup, but it must not silently promote
        model-specific facts or publication status.  Manual mappings become
        ``confirmed`` only when an explicit ``approved_by`` is supplied.
        """
        engineering = engineering_model.strip()
        sales = self._unique_labels(sales_models)
        parent = parent_model.strip()
        if not engineering:
            raise ValueError("engineering_model must not be empty")
        if not sales:
            raise ValueError("sales_models must contain at least one label")
        if provenance not in {"manual", "source"}:
            raise ValueError("provenance must be 'manual' or 'source'")
        evidence = self._validate_mapping_sources(source_uris or [])
        labels = [engineering, *sales, parent]
        resolved_device_uri = device_uri.strip() or self._find_device_uri(labels)
        if resolved_device_uri:
            identity = self.store.lookup_by_uri(resolved_device_uri)
            if identity is None or identity[0] != "Device":
                raise ValueError(f"device_uri must resolve to a Device page: {resolved_device_uri}")
            if self.read_resource(resolved_device_uri) is None:
                raise ValueError(
                    f"device_uri does not point to a readable Device page: {resolved_device_uri}"
                )
        status = "confirmed" if evidence and provenance == "source" else "pending"
        if approved_by.strip():
            status = "confirmed"
        now = datetime.now(UTC).isoformat()
        key = self._model_label(engineering)
        records = self._read_model_mapping_records()
        record = {
            "mapping_id": f"model-map-{key}",
            "engineering_model": engineering,
            "sales_models": sales,
            "parent_model": parent,
            "device_uri": resolved_device_uri,
            "source_uris": evidence,
            "provenance": provenance,
            "status": status,
            "approved_by": approved_by.strip(),
            "notes": notes.strip(),
            "updated_at": now,
        }
        replaced = False
        for index, existing in enumerate(records):
            if self._model_label(str(existing.get("engineering_model", ""))) == key:
                records[index] = record
                replaced = True
                break
        if not replaced:
            records.append(record)
        self.store.write_json(
            self._MODEL_MAPPING_KEY,
            {
                "version": 1,
                "updated_at": now,
                "mappings": sorted(records, key=lambda item: str(item.get("mapping_id", ""))),
            },
        )
        return {**record, "requires_review": status != "confirmed", "replaced": replaced}

    def get_model_mappings(
        self,
        model_id: str = "",
        *,
        include_pending: bool = True,
    ) -> list[dict[str, object]]:
        """Return mappings matching an engineering, sales, or parent label."""
        query = self._model_label(model_id)
        result: list[dict[str, object]] = []
        for record in self._read_model_mapping_records():
            if not include_pending and str(record.get("status", "")) != "confirmed":
                continue
            if query:
                labels = [
                    str(record.get("engineering_model", "")),
                    str(record.get("parent_model", "")),
                    *[
                        str(value)
                        for value in record.get("sales_models", [])
                        if isinstance(value, str)
                    ],
                ]
                if query not in {self._model_label(label) for label in labels}:
                    continue
            result.append(record)
        return result

    def model_mapping_report(self) -> dict[str, object]:
        """Return counts and unresolved mapping records for build reporting."""
        records = self._read_model_mapping_records()
        source_backed = [record for record in records if record.get("source_uris")]
        pending = [record for record in records if record.get("status") != "confirmed"]
        unresolved_device = [
            record for record in records if not str(record.get("device_uri", "")).strip()
        ]
        return {
            "mapping_count": len(records),
            "confirmed_count": len(records) - len(pending),
            "pending_count": len(pending),
            "source_backed_count": len(source_backed),
            "unresolved_device_count": len(unresolved_device),
            "manual_pending_count": sum(
                1 for record in pending if record.get("provenance") == "manual"
            ),
            "records": records,
        }
