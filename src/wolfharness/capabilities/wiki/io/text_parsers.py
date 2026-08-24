"""Pure-function helpers and the TextParsersMixin for WikiBuildTools.

Extracted from ``mcp_server.py`` to keep the composition root small.  All
methods here are ``@staticmethod`` / ``@classmethod`` and have zero ``self``
dependencies, so the mixin can be reused without instantiating
``WikiBuildTools``.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from wolfharness.capabilities.wiki.quality import extract_source_uris


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


logger = logging.getLogger(__name__)


# ── Regex helpers (moved from old builder) ──────────────────────────────────

_CHAPTER_PREFIX_RE = re.compile(r"^\d+_\d+(?:\.\d+)*\s+")
_TITLE_SEGMENT_RE = re.compile(r"^\d+[_\s]*")
_TOC_TITLE_PREFIX_RE = re.compile(r"^\d+(?:\.\d+)*\s+")
_DIR_PREFIX_RE = re.compile(r"^\d+_")

# Matches agent-generated entity headings still carrying a scheme URI (raw
# extraction may emit a path-shaped URI instead of the canonical hash URI);
# write paths normalize it to ``# <canonical_uri>``.
_ENTITY_URI_HEADING_RE = re.compile(r"^#\s+(?:viking://|file://)\S+(?:\r?\n|$)")

# ── Directory-name ↔ TOC-title converters ───────────────────────────────────


def _dir_to_toc_title(dir_name: str) -> str:
    """Convert a chapter directory name to a TOC display title.

    ``01_1 前言`` → ``1 前言``
    ``01_1.1 阅读维修手册的方法`` → ``1.1 阅读维修手册的方法``
    ``02_5.2 液压子系统原理`` → ``5.2 液压子系统原理``
    """
    stripped = _DIR_PREFIX_RE.sub("", dir_name)
    return stripped.strip()


def _dir_to_clean_title(dir_name: str) -> str:
    """Extract a clean (number-free) title from a chapter directory name.

    ``01_1 前言`` → ``前言``
    ``01_1.1 阅读维修手册的方法`` → ``阅读维修手册的方法``
    """
    toc_title = _dir_to_toc_title(dir_name)
    return _TOC_TITLE_PREFIX_RE.sub("", toc_title).strip()


def _parse_forward_links(content: str) -> list[str]:
    """Extract all source URIs from content (forward links)."""
    return sorted(extract_source_uris(content))


class TextParsersMixin:
    """Pure parsing helpers shared by :class:`WikiBuildTools`."""

    _H2_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)

    _FM_BOUNDARY_RE = re.compile(r"\n---\n")

    @staticmethod
    def _build_chapter_map(
        doc_id: str,
        chapters: list[dict],
        make_uri: Callable[[str, str], str],
    ) -> dict[str, str]:
        """Build title → raw chapter URI dict from chapter list.

        Primary keys (in order):
          1. **Clean TOC title** — number prefix stripped (``发动机冒黑烟``)
          2. **Raw TOC title** — as-is from ``toc.json`` (``9.9.6 发动机冒黑烟``)
          3. **Subdir-derived title** — last path segment, prefix stripped

        Dual indexing lets agents look up by either form: the clean title
        extracted from ``toc.md`` or the raw line from the TOC markdown.
        When multiple chapters share a key, the first wins.

        ``make_uri`` builds the real-path URI for a ``(doc_id, subdir)``
        pair; the mixin wires it to ``self.make_source_uri``.
        """
        mapping: dict[str, str] = {}
        for ch in chapters:
            subdir = ch.get("subdir", "")
            if not subdir:
                continue
            uri = make_uri(doc_id, subdir)

            title = ch.get("title", "")
            if title:
                raw = title.strip()
                clean = _TOC_TITLE_PREFIX_RE.sub("", raw).strip()
                if clean and clean not in mapping:
                    mapping[clean] = uri
                if raw != clean and raw not in mapping:
                    mapping[raw] = uri

            last_seg = subdir.rsplit("/", 1)[-1]
            derived = _TITLE_SEGMENT_RE.sub("", last_seg).strip()
            if derived and derived not in mapping:
                mapping[derived] = uri
            if subdir not in mapping:
                mapping[subdir] = uri

        return mapping

    @staticmethod
    def _extract_model_from_doc(doc_dir: Path, pdf_path: str) -> str:
        """Extract model ID from filename first, then from parsed content.

        Phase 1: regex on filename / doc_id (fastest).
        Phase 2: regex on first 200 lines of ``full.md`` (parsed PDF content).
        Falls back to ``"unknown"`` if both fail.
        """
        # Phase 1: filename
        for p in (pdf_path, doc_dir.name):
            m = re.search(r"([A-Z]+[0-9]+[A-Z]?)", p)
            if m:
                return m.group(1).lower()

        # Phase 2: parsed content (full.md)
        full_md = doc_dir / "full.md"
        if full_md.is_file():
            try:
                with full_md.open(encoding="utf-8") as f:
                    for _i, line in enumerate(f):
                        if _i >= 200:
                            break
                        m = re.search(r"\b([A-Z]{2,}[0-9]+[A-Z]?)\b", line)
                        if m:
                            candidate = m.group(1)
                            # Filter out non-model matches like "PDF", "ISBN"
                            if candidate not in {
                                "PDF",
                                "ISBN",
                                "URL",
                                "HTTP",
                                "HTML",
                                "XML",
                                "JSON",
                            }:
                                return candidate.lower()
            except OSError:
                pass

        logger.warning("Could not extract model from %s (pdf_path=%s)", doc_dir.name, pdf_path)
        return "unknown"

    @classmethod
    def _dedupe_h2_sections(cls, content: str) -> str:
        """Merge repeated level-two sections without losing their evidence.

        Relation workers use section patches, and an interrupted/retried patch
        can otherwise append a second copy of the same heading.  Keeping the
        first heading and merging only non-duplicate section bodies makes the
        operation idempotent and preserves source text.
        """
        if not content.startswith("---\n"):
            body_start = 0
        else:
            closing = content.find("\n---\n", 4)
            if closing == -1:
                return content
            body_start = closing + 5
        prefix = content[:body_start]
        body = content[body_start:]
        matches = list(cls._H2_SECTION_RE.finditer(body))
        if len(matches) < 2:
            return content

        preamble = body[: matches[0].start()]
        sections: list[tuple[str, str]] = []
        positions: dict[str, int] = {}
        for index, match in enumerate(matches):
            heading = match.group(1).strip()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
            section_body = body[match.end() : end].strip()
            existing_index = positions.get(heading)
            if existing_index is None:
                positions[heading] = len(sections)
                sections.append((heading, section_body))
                continue
            old_heading, old_body = sections[existing_index]
            if not section_body:
                continue
            old_normalized = re.sub(r"\s+", " ", old_body).strip()
            new_normalized = re.sub(r"\s+", " ", section_body).strip()
            if new_normalized and new_normalized not in old_normalized:
                merged = f"{old_body}\n\n{section_body}" if old_body else section_body
                sections[existing_index] = (old_heading, merged)

        rendered = [preamble]
        for heading, section_body in sections:
            rendered.append(f"## {heading}\n")
            if section_body:
                rendered.append(f"{section_body}\n")
        return prefix + "".join(rendered)

    @staticmethod
    def _frontmatter_uri_values(
        root_uri: str, frontmatter: dict[str, object], field: str
    ) -> list[str]:
        """Return URI-like values from a YAML relation field."""
        prefix = root_uri + "/"
        value = frontmatter.get(field)
        if isinstance(value, str):
            return [value] if value.startswith(prefix) else []
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str) and item.startswith(prefix)]

    @classmethod
    def _append_section_links(
        cls,
        content: str,
        heading: str,
        links: list[tuple[str, str]],
    ) -> str:
        """Append canonical Markdown links to one existing ``##`` section."""
        if not links:
            return content
        matches = list(cls._H2_SECTION_RE.finditer(content))
        target_index = next(
            (index for index, match in enumerate(matches) if match.group(1).strip() == heading),
            None,
        )
        link_lines = [f"- [{label}]({uri})" for label, uri in links]
        if target_index is None:
            suffix = "\n" if content.endswith("\n") else "\n\n"
            return f"{content}{suffix}## {heading}\n\n" + "\n".join(link_lines) + "\n"
        match = matches[target_index]
        end = matches[target_index + 1].start() if target_index + 1 < len(matches) else len(content)
        section = content[match.end() : end]
        missing = [line for line in link_lines if line not in section]
        if not missing:
            return content
        insertion = section.rstrip() + "\n" + "\n".join(missing) + "\n"
        return content[: match.end()] + insertion + content[end:]

    @staticmethod
    def _apply_line_op(content: str, op: dict) -> str:
        """Apply a line-based patch operation (1-indexed)."""
        lines = content.splitlines(keepends=True)

        if op["op"] == "line_replace":
            start, end = op["start"], op["end"]
            if start < 1 or end > len(lines) or start > end:
                raise ValueError(
                    f"line_replace range [{start}, {end}] out of bounds (1..{len(lines)})"
                )
            replacement = op["content"]
            if not replacement.endswith("\n"):
                replacement += "\n"
            lines[start - 1 : end] = replacement.splitlines(keepends=True)

        elif op["op"] == "line_insert":
            at = op["at"]
            if at < 1 or at > len(lines) + 1:
                raise ValueError(f"line_insert position {at} out of bounds (1..{len(lines) + 1})")
            insertion = op["content"]
            if not insertion.endswith("\n"):
                insertion += "\n"
            lines[at - 1 : at - 1] = insertion.splitlines(keepends=True)

        elif op["op"] == "line_delete":
            start, end = op["start"], op["end"]
            if start < 1 or end > len(lines) or start > end:
                raise ValueError(
                    f"line_delete range [{start}, {end}] out of bounds (1..{len(lines)})"
                )
            del lines[start - 1 : end]

        return "".join(lines)

    @staticmethod
    def _apply_section_op(content: str, op: dict) -> str:
        """Apply a section-based patch operation.

        Sections are identified by ``## <heading>`` lines.  A section spans
        from its heading line to the next ``## `` heading or end of body.
        """
        # Split into heading + frontmatter + body
        heading_end = content.find("\n")
        if heading_end == -1:
            raise ValueError("Content has no body — cannot patch sections")

        rest = content[heading_end + 1 :]

        # Check for frontmatter
        fm_text = ""
        body_start = 0
        if rest.startswith("---\n"):
            fm_close = rest.find("\n---\n", 4)
            if fm_close != -1:
                fm_text = rest[: fm_close + 5]
                body_start = fm_close + 5

        body = rest[body_start:]
        body_lines = body.splitlines(keepends=True)

        # Find section boundaries: list of (heading_text, line_index) for ## headings
        section_starts: list[tuple[str, int]] = []
        for i, line in enumerate(body_lines):
            if line.startswith("## "):
                heading_text = line[3:].strip()
                section_starts.append((heading_text, i))

        target_heading = op["heading"]
        target_idx = None
        for idx, (h, _line_index) in enumerate(section_starts):
            if h == target_heading:
                target_idx = idx
                break
        if target_idx is None:
            raise ValueError(f"Section not found: {target_heading!r}")

        section_start_line = section_starts[target_idx][1]
        if target_idx + 1 < len(section_starts):
            section_end_line = section_starts[target_idx + 1][1]
        else:
            section_end_line = len(body_lines)

        if op["op"] == "section_replace":
            new_content = op["content"]
            # ponytail: auto-prepend heading to prevent accidental heading deletion
            if not new_content.lstrip().startswith(f"## {target_heading}"):
                new_content = f"## {target_heading}\n{new_content}"
            if not new_content.endswith("\n"):
                new_content += "\n"
            new_lines = new_content.splitlines(keepends=True)
            body_lines[section_start_line:section_end_line] = new_lines

        elif op["op"] == "section_insert_after":
            heading_new = op["heading_new"]
            new_content = op["content"]
            block = f"## {heading_new}\n"
            if not new_content.endswith("\n"):
                new_content += "\n"
            block += new_content
            body_lines[section_end_line:section_end_line] = block.splitlines(keepends=True)

        new_body = "".join(body_lines)
        return content[: heading_end + 1] + fm_text + new_body

    @staticmethod
    def _apply_fm_op(content: str, op: dict) -> str:
        """Apply a frontmatter field patch operation."""
        # Locate frontmatter boundaries
        fm_start = content.find("---\n")
        if fm_start == -1:
            raise ValueError("No frontmatter found in entity content")
        fm_end = content.find("\n---\n", fm_start + 4)
        if fm_end == -1:
            raise ValueError("Malformed frontmatter — no closing ---")

        fm_end += 4  # include the closing "---\n"
        fm_text = content[:fm_end]
        body = content[fm_end:]
        fm_lines = fm_text.splitlines(keepends=True)

        field = op["field"]

        if op["op"] == "fm_set":
            # Find existing field or append
            found = False
            for i, line in enumerate(fm_lines):
                if line.strip().startswith(f"{field}:") and not line.startswith(" "):
                    fm_lines[i] = f"{field}: {op['value']}\n"
                    found = True
                    break
            if not found:
                # Insert before closing ---
                fm_lines.insert(-1, f"{field}: {op['value']}\n")

        elif op["op"] == "fm_set_list":
            values = op["values"]
            field_line_idx = None
            for i, line in enumerate(fm_lines):
                if line.strip().startswith(f"{field}:") and not line.startswith(" "):
                    field_line_idx = i
                    break
            if field_line_idx is not None:
                item_end = field_line_idx + 1
                for j in range(field_line_idx + 1, len(fm_lines)):
                    stripped = fm_lines[j].rstrip()
                    if stripped.startswith("  - "):
                        item_end = j + 1
                    else:
                        break
                fm_lines[field_line_idx] = f"{field}:\n"
                del fm_lines[field_line_idx + 1 : item_end]
                new_lines = [f"  - {v}\n" for v in values]
                fm_lines[field_line_idx + 1 : field_line_idx + 1] = new_lines
            else:
                insert_block = [f"{field}:\n"]
                insert_block.extend(f"  - {v}\n" for v in values)
                fm_lines.insert(-1, "".join(insert_block))

        elif op["op"] == "fm_append":
            values = op["values"]
            # Find existing field
            field_line_idx = None
            for i, line in enumerate(fm_lines):
                if line.strip() == f"{field}:" or line.strip().startswith(f"{field}: []"):
                    field_line_idx = i
                    break
                if line.strip().startswith(f"{field}:") and not line.startswith(" "):
                    # Inline list field like "field: [a, b]"
                    field_line_idx = i
                    break

            if field_line_idx is not None:
                # Find existing list items under this field
                existing_items: list[str] = []
                item_start = field_line_idx + 1
                item_end = item_start
                for j in range(item_start, len(fm_lines)):
                    stripped = fm_lines[j].rstrip()
                    if stripped.startswith("  - "):
                        existing_items.append(stripped[4:].strip())
                        item_end = j + 1
                    else:
                        break

                # Check if it's an inline list
                field_line = fm_lines[field_line_idx].strip()
                if field_line != f"{field}:" and not field_line.startswith(f"{field}: []"):
                    # Parse inline list
                    bracket_content = field_line[field_line.find("[") + 1 : field_line.rfind("]")]
                    if bracket_content.strip():
                        existing_items.extend(v.strip() for v in bracket_content.split(","))
                    # Convert to block list format, re-inserting existing items
                    fm_lines[field_line_idx] = f"{field}:\n"
                    existing_item_lines = [f"  - {v}\n" for v in existing_items]
                    fm_lines[field_line_idx + 1 : field_line_idx + 1] = existing_item_lines
                    item_start = field_line_idx + 1
                    item_end = item_start + len(existing_item_lines)
                elif field_line.startswith(f"{field}: []"):
                    # Empty list `[]` — convert to block list format before appending
                    fm_lines[field_line_idx] = f"{field}:\n"
                    item_start = field_line_idx + 1
                    item_end = item_start

                # Append new values (dedup)
                existing_set = set(existing_items)
                new_items: list[str] = []
                for v in values:
                    if v not in existing_set:
                        existing_set.add(v)
                        new_items.append(v)

                if new_items:
                    new_item_lines = [f"  - {v}\n" for v in new_items]
                    fm_lines[item_end:item_end] = new_item_lines
            else:
                # Field doesn't exist — add it before closing ---
                insert_block = [f"{field}:\n"]
                insert_block.extend(f"  - {v}\n" for v in values)
                fm_lines.insert(-1, "".join(insert_block))

        new_fm = "".join(fm_lines)
        return new_fm + body

    @staticmethod
    def _read_markdown_fragment(content: str, fragment: str) -> str | None:
        """Return the Markdown section(s) addressed by a fragment.

        Generated references sometimes carry a display qualifier after the
        heading (for example ``SPEC-输出电压（B19``) or combine adjacent
        specification headings with ``/``. Resolve those forms against the
        actual heading text while still returning ``None`` for an unrelated
        fragment.
        """
        if not fragment:
            return content
        heading_pattern = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$", re.MULTILINE)
        headings = list(heading_pattern.finditer(content))
        requested = [part.strip() for part in fragment.split("/") if part.strip()]
        selected: list[re.Match[str]] = []
        for part in requested:
            match = next(
                (
                    candidate
                    for candidate in headings
                    if candidate.group("title") == part
                    or any(
                        part.startswith(candidate.group("title") + prefix)
                        for prefix in ("（", " ", "—")
                    )
                    or any(
                        candidate.group("title").startswith(part + prefix)
                        for prefix in ("（", " ", "—")
                    )
                ),
                None,
            )
            if match is not None and match not in selected:
                selected.append(match)
        if not selected:
            return None
        sections: list[str] = []
        for match in selected:
            level = len(match.group("level"))
            following_heading = re.compile(rf"^#{{1,{level}}}\s+", re.MULTILINE)
            next_match = following_heading.search(content, match.end())
            end = next_match.start() if next_match is not None else len(content)
            sections.append(content[match.start() : end].rstrip())
        return "\n\n".join(sections) + "\n"

    @staticmethod
    def _read_legacy_text(root: Path, relative: str) -> str | None:
        resolved_root = root.resolve()
        requested = (resolved_root / relative).resolve()
        for path in (requested, requested.with_suffix(".md")):
            if path.is_relative_to(resolved_root) and path.is_file() and not path.is_symlink():
                return path.read_text(encoding="utf-8")
        return None
