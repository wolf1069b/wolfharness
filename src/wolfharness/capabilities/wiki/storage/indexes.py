"""Read-only compatibility for identity maps written by older wiki builds.

New entities and classes use readable, self-describing paths. This module is
deliberately not a write path: it only keeps legacy hash URIs and historical
case aliases readable during rolling migration.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .backend import FSBackend


WikiIdentity = tuple[str, str | None, str]


class LegacyWikiIndex:
    """Resolve historical JSON mappings without extending them."""

    def __init__(self, backend: FSBackend) -> None:
        self._backend = backend

    def _read_mapping(self, name: str) -> dict[str, str]:
        key = f"index/{name}"
        raw = self._backend.read_text(key)
        if raw is None:
            return {}
        data = json.loads(raw)
        if not isinstance(data, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in data.items()
        ):
            raise ValueError(f"Invalid legacy wiki mapping: {key}")
        return data

    @staticmethod
    def _parse_natural_key(key: str) -> WikiIdentity | None:
        parts = key.split(":", 2)
        if len(parts) != 3:
            return None
        concept, class_name, object_name = parts
        return concept, class_name or None, object_name

    def lookup_uri(self, uri: str) -> WikiIdentity | None:
        """Resolve an old entity hash URI through read-only legacy mappings."""
        reverse = self._read_mapping("uri_keys.json")
        natural_key = reverse.get(uri)
        if natural_key is None:
            natural_keys = self._read_mapping("natural_keys.json")
            natural_key = next(
                (key for key, candidate_uri in natural_keys.items() if candidate_uri == uri),
                None,
            )
        return self._parse_natural_key(natural_key) if natural_key is not None else None

    def lookup_class_uri(self, uri: str) -> tuple[str, str] | None:
        """Resolve an old class hash URI through its read-only mapping."""
        prefix = self._backend.root_uri + "/"
        if not uri.startswith(prefix):
            return None
        parts = uri[len(prefix) :].strip("/").split("/")
        if len(parts) < 2:
            return None
        entry = self._read_mapping("class_keys.json").get(parts[-1])
        if entry is None or ":" not in entry:
            return None
        concept, class_name = entry.split(":", 1)
        return concept, class_name

    def resolve_case_uri(self, uri: str) -> str | None:
        """Resolve either side of a historical case URI alias."""
        return self._read_mapping("case_uri_map.json").get(uri)


# Import compatibility for integrations that referenced the old class name.
WikiIndex = LegacyWikiIndex
