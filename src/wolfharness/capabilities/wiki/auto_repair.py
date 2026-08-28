"""Deterministic post-processing repairs for wiki entities.

These functions handle the mechanical fixes that don't require LLM reasoning:
- ``[open_gap]`` placeholder cleanup in frontmatter
- body-link materialization from frontmatter relation fields

The LLM agents (conductor, relation_worker, file_operator) call
``batch_auto_repair`` once after extraction; only the remaining issues
that truly need reasoning (merge decisions, dangling-reference repair)
are left for agent iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import TYPE_CHECKING

from wolfharness.capabilities.wiki.quality import (
    _BODY_LINK_MAP,
    _is_profile,
    extract_sections,
    extract_wiki_uris,
    parse_frontmatter,
)


if TYPE_CHECKING:
    from wolfharness.capabilities.wiki.wiki_build_tools import WikiBuildTools

logger = logging.getLogger(__name__)

# ── open_gap cleanup ──────────────────────────────────────────────────────

_OPEN_GAP_INLINE_RE = re.compile(r":\s*\[open_gap\]\s*$", re.MULTILINE)
_OPEN_GAP_LIST_RE = re.compile(
    r"^(\s+)-\s+open_gap\s*$",
    re.MULTILINE,
)


def clean_open_gap(content: str) -> str:
    """Replace ``[open_gap]`` placeholders with empty lists.

    Handles two YAML patterns:
    - Inline: ``field: [open_gap]`` → ``field: []``
    - List item: ``  - open_gap`` → removed (field becomes empty list)

    Does NOT touch ``open_gap`` appearing in body prose — that is
    source-honest gap text that the agent should decide about.
    """
    # Inline: field: [open_gap]
    content = _OPEN_GAP_INLINE_RE.sub(": []", content)
    # List item: "  - open_gap" inside a YAML list
    # Remove the line; if the parent field now has no items, it stays
    # as a bare "field:" which YAML parses as None → treated as empty.
    content = _OPEN_GAP_LIST_RE.sub("", content)
    # Clean up any double blank lines left by list-item removal
    return re.sub(r"\n{3,}", "\n\n", content)


# ── body-link materialization ─────────────────────────────────────────────


def _frontmatter_uri_list(frontmatter: dict[str, object], field: str) -> list[str]:
    """Extract URI strings from a frontmatter list field."""
    value = frontmatter.get(field)
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def _resolve_uri_label(tools: WikiBuildTools, uri: str) -> str:
    """Look up an entity's title/object_name for human-readable link text."""
    info = tools.store.lookup_by_uri(uri)
    if info is None:
        return uri
    _concept, _class_name, object_name = info
    content = tools.store.read_entity_by_uri(uri)
    if content is not None:
        fm = parse_frontmatter(content)
        title = fm.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()
    return object_name


def materialize_body_links(
    content: str,
    concept: str,
    tools: WikiBuildTools,
    *,
    known_uris: set[str] | None = None,
) -> str:
    """Append frontmatter relation URIs to body sections if missing.

    For each relation field in the body-link map, reads the URIs from
    frontmatter, checks which are already linked in the corresponding
    body section, and appends the missing ones as Markdown links.

    Returns the updated content (or original if no changes needed).
    """
    is_profile = _is_profile(content)
    link_map_key = "_Profile" if is_profile else concept
    field_map = _BODY_LINK_MAP.get(link_map_key, {})
    if not field_map:
        return content

    frontmatter = parse_frontmatter(content)
    sections = extract_sections(content)
    changes: list[tuple[str, list[str]]] = []  # (heading, missing_uri_lines)

    for fm_field, (heading, _target_concept) in field_map.items():
        uris = _frontmatter_uri_list(frontmatter, fm_field)
        if not uris:
            continue
        existing_section_uris = extract_wiki_uris(sections.get(heading, ""))
        missing: list[str] = []
        for uri in uris:
            canonical = uri.split("#", 1)[0]
            if canonical not in existing_section_uris:
                label = _resolve_uri_label(tools, canonical)
                missing.append(f"- [{label}]({canonical})")
        if missing:
            changes.append((heading, missing))

    if not changes:
        return content

    # Apply changes by inserting into or creating sections
    for heading, missing_lines in changes:
        content = _append_to_section(content, heading, missing_lines)

    return content


def _append_to_section(content: str, heading: str, lines: list[str]) -> str:
    """Append link lines to a ``##`` section, creating it if absent."""
    h2_re = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
    matches = list(h2_re.finditer(content))
    target_idx = next(
        (i for i, m in enumerate(matches) if m.group(1).strip() == heading),
        None,
    )
    if target_idx is None:
        # Create section at end of body
        suffix = "\n" if content.endswith("\n") else "\n\n"
        return f"{content}{suffix}## {heading}\n\n" + "\n".join(lines) + "\n"
    match = matches[target_idx]
    end = matches[target_idx + 1].start() if target_idx + 1 < len(matches) else len(content)
    section = content[match.end() : end]
    # Check each line isn't already present
    truly_missing = [line for line in lines if line not in section]
    if not truly_missing:
        return content
    insertion = section.rstrip() + "\n" + "\n".join(truly_missing) + "\n"
    return content[: match.end()] + insertion + content[end:]


# ── batch repair ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RepairReport:
    """Summary of deterministic repairs applied to the wiki corpus."""

    entities_scanned: int = 0
    open_gap_cleaned: int = 0
    body_links_added: int = 0
    entities_modified: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entities_scanned": self.entities_scanned,
            "open_gap_cleaned": self.open_gap_cleaned,
            "body_links_added": self.body_links_added,
            "entities_modified": self.entities_modified,
            "errors": self.errors,
        }


def batch_auto_repair(tools: WikiBuildTools, entity_uris: list[str] | None = None) -> RepairReport:
    """Run all deterministic repairs on every formal entity in the wiki.

    Iterates all entities from the natural-keys index (including Symptom
    Profile sub-resources), applies ``clean_open_gap`` and
    ``materialize_body_links`` in-place, and writes back only changed files.

    When *entity_uris* is provided, only entities whose URI is in that set
    are scanned — used by the conductor to scope repair to the current build.

    Returns a :class:`RepairReport` summarising the work.
    """
    scoped_set: set[str] | None = set(entity_uris) if entity_uris else None
    errors: list[str] = []
    entities_scanned = 0
    open_gap_total = 0
    body_links_total = 0
    entities_modified = 0

    for concept, class_name, object_name, uri in tools.store.list_entities():
        if scoped_set is not None and uri not in scoped_set:
            continue
        entities_scanned += 1
        if class_name is None:
            continue
        try:
            content = tools.store.read_entity_by_uri(uri)
            if content is None:
                continue

            original = content

            content = clean_open_gap(content)
            if content != original:
                open_gap_total += 1

            content_before_links = content
            content = materialize_body_links(content, concept, tools)
            if content != content_before_links:
                body_links_total += 1

            if content != original:
                diff = tools.diff_entity(concept, class_name, object_name, content)
                if bool(diff["changed"]):
                    current_sha256 = diff["current_sha256"]
                    if not isinstance(current_sha256, str):
                        raise TypeError("diff_entity returned a non-string current_sha256")
                    tools.merge_entity(
                        concept,
                        class_name,
                        object_name,
                        content,
                        expected_sha256=current_sha256,
                        skip_materialization=True,  # ponytail: auto_repair only adds body links/cleans gaps; don't re-validate full page
                    )
                entities_modified += 1
                logger.info("Auto-repaired: %s (%s/%s)", uri, concept, object_name)

            # Process Symptom Profile sub-resources
            if concept == "Symptom":
                for profile in tools.list_symptom_profiles(uri):
                    profile_id = profile["profile_id"]
                    profile_uri = profile["uri"]
                    entities_scanned += 1
                    try:
                        profile_content = tools.read_resource(profile_uri)
                        if profile_content is None:
                            continue
                        profile_original = profile_content

                        profile_content = clean_open_gap(profile_content)
                        if profile_content != profile_original:
                            open_gap_total += 1

                        profile_before_links = profile_content
                        profile_content = materialize_body_links(profile_content, "Symptom", tools)
                        if profile_content != profile_before_links:
                            body_links_total += 1

                        if profile_content != profile_original:
                            diff = tools.diff_symptom_profile(uri, profile_id, profile_content)
                            if bool(diff["changed"]):
                                current_sha256 = diff["current_sha256"]
                                if not isinstance(current_sha256, str):
                                    raise TypeError(
                                        "diff_symptom_profile returned a non-string current_sha256",
                                    )
                                # ``write_symptom_profile`` is the guarded merge
                                # primitive for Profile resources: existing
                                # profiles require the hash returned by diff.
                                tools.write_symptom_profile(
                                    uri,
                                    profile_id,
                                    profile_content,
                                    expected_sha256=current_sha256,
                                )
                            entities_modified += 1
                            logger.info("Auto-repaired profile: %s", profile_uri)
                    except (OSError, TypeError, ValueError, KeyError) as exc:
                        errors.append(f"{profile_uri}: {exc!s}")
                        logger.warning("Auto-repair error for %s: %s", profile_uri, exc)

        except (OSError, TypeError, ValueError, KeyError) as exc:
            errors.append(f"{uri}: {exc!s}")
            logger.warning("Auto-repair error for %s: %s", uri, exc)

    return RepairReport(
        entities_scanned=entities_scanned,
        open_gap_cleaned=open_gap_total,
        body_links_added=body_links_total,
        entities_modified=entities_modified,
        errors=errors,
    )
