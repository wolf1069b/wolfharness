"""Validate BOM-driven Component identity and logical classification paths."""

from __future__ import annotations

from wolfharness.capabilities.wiki.quality import (
    is_bom_source_uri,
    is_external_source_uri,
    is_raw_chapter_uri,
    parse_frontmatter,
    wiki_uri_prefix,
)
from wolfharness.capabilities.wiki.section_constants import MODEL_TOKEN_RE

from .base import BaseHook, HookResult


_FORBIDDEN_PREFIXES = ("关重件/", "普通件/")


def _string_list(value: object) -> list[str]:
    """Normalize YAML scalar/list values without accepting arbitrary mappings."""
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


class ComponentTaxonomyHook(BaseHook):
    """Keep Component folders as evidence-backed BOM paths, not model buckets."""

    @property
    def name(self) -> str:
        return "component_taxonomy"

    def check(
        self,
        content: str,
        concept: str = "",
        class_name: str = "",
        object_name: str = "",
    ) -> HookResult:
        if concept != "Component":
            return HookResult(self.name, True, "BOM taxonomy applies only to Component entities.")

        frontmatter = parse_frontmatter(content)
        declared_class = frontmatter.get("class_name")
        logical_class = (
            declared_class.strip()
            if isinstance(declared_class, str) and declared_class.strip()
            else class_name
        )
        failures: list[str] = []
        if any(logical_class.startswith(prefix) for prefix in _FORBIDDEN_PREFIXES):
            failures.append("physical tier prefix is forbidden; use the logical BOM path")

        segments = [segment for segment in logical_class.split("/") if segment]
        applicable_models = _string_list(frontmatter.get("applicable_models"))
        if segments and any(
            segments[0].casefold() == model.casefold() for model in applicable_models
        ):
            failures.append("Component class_name must not use a machine model as its root")
        if segments and MODEL_TOKEN_RE.fullmatch(segments[0]) is not None:
            failures.append("Component class_name must not use a machine-model token as its root")

        bom_path = frontmatter.get("bom_path")
        if isinstance(bom_path, str) and bom_path.strip():
            if logical_class != bom_path.strip():
                failures.append("class_name does not match the declared BOM path")
            evidence = _string_list(frontmatter.get("bom_evidence"))
            if not any(
                is_raw_chapter_uri(uri) or is_external_source_uri(uri) or is_bom_source_uri(uri)
                for uri in evidence
            ):
                failures.append("declared BOM path has no raw BOM evidence")
            parent_uri = frontmatter.get("bom_parent_uri")
            if parent_uri is not None and (
                not isinstance(parent_uri, str)
                or not parent_uri.startswith(wiki_uri_prefix() + "/Component/")
            ):
                failures.append("bom_parent_uri must reference a Component")

        if failures:
            return HookResult(
                self.name,
                False,
                f"Component taxonomy validation failed for {object_name or 'unnamed'}: {'; '.join(failures)}.",
                severity="error",
            )
        return HookResult(
            self.name, True, "Component uses a logical, BOM-compatible classification path."
        )
