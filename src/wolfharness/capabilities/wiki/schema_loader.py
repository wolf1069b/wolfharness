"""Cached loader for the wiki entity schema YAML.

Both validation hooks (``frontmatter_schema.py``, ``body_sections.py``) and
the ``get_schema`` build tool share ``templates/default_schema.yaml`` as the
single source of truth.  This module loads it once at first access and
exposes typed accessors so hooks never hardcode field/section lists that
can drift from the YAML.

If the YAML is missing or unreadable, every accessor returns an empty
result so the hooks fall through to their built-in fallbacks.
"""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


_SCHEMA_PATH = Path(__file__).resolve().parent / "templates" / "default_schema.yaml"


@lru_cache(maxsize=1)
def _load_raw() -> dict[str, Any]:
    """Load and cache the raw YAML schema dict.

    Returns an empty dict when the file is missing so callers can
    degrade gracefully.
    """
    if not _SCHEMA_PATH.exists():
        return {}
    try:
        data = yaml.safe_load(_SCHEMA_PATH.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _concepts() -> dict[str, Any]:
    """Return the ``concepts:`` mapping from the schema YAML."""
    raw = _load_raw()
    concepts = raw.get("concepts")
    return concepts if isinstance(concepts, dict) else {}


def get_schema_version() -> str:
    """Return the materialization schema version as a stable string."""
    version = _load_raw().get("version")
    if isinstance(version, (str, int)) and str(version).strip():
        return str(version).strip()
    return "unknown"


def get_concept_schema(concept: str) -> dict[str, Any]:
    """Return one concept's authoritative materialization schema.

    The returned mapping is detached from the process cache so callers may
    serialize or annotate it without mutating validation state.
    """
    entry = _concepts().get(concept)
    if not isinstance(entry, dict):
        supported = ", ".join(sorted(_concepts()))
        raise KeyError(f"Unknown concept {concept!r}; supported concepts: {supported}")
    return copy.deepcopy(entry)


def _shared_frontmatter_fields() -> frozenset[str]:
    """Return orthogonal build/review fields shared by every page type."""
    shared = _load_raw().get("shared_frontmatter")
    if not isinstance(shared, list):
        return frozenset()
    return frozenset(
        item["name"]
        for item in shared
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    )


def get_concept_frontmatter_fields(concept: str) -> frozenset[str]:
    """Return the set of allowed frontmatter field names for *concept*.

    Returns an empty set when the concept or its ``frontmatter`` key is
    absent from the YAML.
    """
    entry = _concepts().get(concept)
    if not isinstance(entry, dict):
        return frozenset()
    fm = entry.get("frontmatter")
    if not isinstance(fm, list):
        return frozenset()
    fields = frozenset(item["name"] for item in fm if isinstance(item, dict) and "name" in item)
    return fields | _shared_frontmatter_fields()


def get_all_concept_frontmatter_fields() -> dict[str, frozenset[str]]:
    """Return ``{concept: frozenset(field_names)}`` for all concepts."""
    return {name: get_concept_frontmatter_fields(name) for name in _concepts()}


def get_profile_frontmatter_fields() -> frozenset[str]:
    """Return frontmatter fields for Symptom Profile sub-resources.

    Looks up ``concepts.Symptom.sub_resources.profiles.frontmatter``.
    """
    symptom = _concepts().get("Symptom")
    if not isinstance(symptom, dict):
        return frozenset()
    sub = symptom.get("sub_resources")
    if not isinstance(sub, dict):
        return frozenset()
    profiles = sub.get("profiles")
    if not isinstance(profiles, dict):
        return frozenset()
    fm = profiles.get("frontmatter")
    if not isinstance(fm, list):
        return frozenset()
    fields = frozenset(item["name"] for item in fm if isinstance(item, dict) and "name" in item)
    return fields | _shared_frontmatter_fields()


def get_concept_body_sections(concept: str) -> frozenset[str]:
    """Return the set of required body section titles for *concept*."""
    entry = _concepts().get(concept)
    if not isinstance(entry, dict):
        return frozenset()
    sections = entry.get("body_sections")
    if not isinstance(sections, list):
        return frozenset()
    return frozenset(str(s) for s in sections)


def get_all_concept_body_sections() -> dict[str, frozenset[str]]:
    """Return ``{concept: frozenset(section_titles)}`` for all concepts."""
    return {name: get_concept_body_sections(name) for name in _concepts()}


def get_profile_body_sections() -> frozenset[str]:
    """Return body sections for Symptom Profile sub-resources."""
    symptom = _concepts().get("Symptom")
    if not isinstance(symptom, dict):
        return frozenset()
    sub = symptom.get("sub_resources")
    if not isinstance(sub, dict):
        return frozenset()
    profiles = sub.get("profiles")
    if not isinstance(profiles, dict):
        return frozenset()
    sections = profiles.get("body_sections")
    if not isinstance(sections, list):
        return frozenset()
    return frozenset(str(s) for s in sections)
