"""Compact raw chapter text before it reaches the model context.

Wiki chapters are OCR-converted PDFs: they carry a lot of boilerplate that
carries zero entity-extraction signal — HTML residue (``<td>``/``<div>``),
image placeholders (``<image_token …>``), and verbose table rows.  Repeating
template lines appear in 1 of every 4 chapters, so stripping them at the read
boundary cuts model input tokens with no loss of extractable facts.

The compaction is applied only at the *read* boundary (``read_chapter`` /
``read_raw_resource``) — the raw files and their hashes are untouched, so
source/evidence references stay valid.  It is intentionally lossy only in
layout: run values, units, part numbers, and DTC codes are all preserved.

This module is pure and framework-free so it can be unit tested directly.
"""

from __future__ import annotations

import re


# Table rows and image placeholders contribute noise, not entities.
_TABLE_TAG_RE = re.compile(r"</?(?:td|th|tr|table)\b[^>]*>", flags=re.IGNORECASE)
_IMAGE_TOKEN_RE = re.compile(r"<image_token\b[^>]*/?>", flags=re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[a-zA-Z/][^>]*>")
# Footnote divs carry part annotations (figure labels like "A1. 至行走马达PT油口").
# Strip the wrapper tags but keep the inner text — the annotations are entity
# evidence, not boilerplate.
_FOOTNOTE_OPEN_RE = re.compile(r"<div\b[^>]*class=\"footnote\"[^>]*>\s*", flags=re.IGNORECASE)
_FOOTNOTE_CLOSE_RE = re.compile(r"</div>\s*", flags=re.IGNORECASE)
_EMPTY_CELL_RE = re.compile(
    r"<\s*(?:td|th|tr|table)\s*([^>]*?)\s*>\s*</\s*(?:td|th|tr|table)\s*>", flags=re.IGNORECASE
)

# Rows that repeat verbatim across every chapter (safety notices, torque
# reminders).  Dropping them removes up to ~26% of duplicated input.
# Star-prefixed lines only drop when they carry NO measurement fingerprint
# (units/numbers), so "★ 使用 6 MPa 油压计" survives while fixed template
# noise like "★ 安装过程中螺栓扭矩…" is removed.
_HAS_NUMERIC_FINGERPRINT = re.compile(
    r"\d+(?:\.\d+)?\s*(?:MPa|kPa|kgf|kg|N·m|V|Ω|A|rpm|mm|bar|℃|°C|L/min)",
)
_NOISE_LINE_RE = re.compile(
    r"^\s*(?:"
    r"[-*]\s*(?:安装|装配|拆卸|维修|保养)过程中螺栓扭矩[^\n]*|"
    r"注意[：:][^\n]*|"
    r"警告[：:][^\n]*"
    r")\s*$",
)
_EMPTY_TABLE_ROW_RE = re.compile(
    r"^\s*(?:<td[^>]*>\s*</td>|<td colspan=[\"']?\d+[\"']?\s*</td>\s*)+$"
)


def compact_chapter(content: str) -> str:
    """Return a layout-compacted copy of a raw chapter for model reading.

    Keeps all extractable text: prose, tables (stripped of HTML tags but
    cells intact), part numbers, DTC codes, run values.  Only removes
    HTML scaffolding, image placeholders, and cross-chapter boilerplate
    lines.

    Args:
        content: Full raw chapter markdown/HTML text.

    Returns:
        Compacted text with the same newline structure where meaningful.
    """
    text = content
    text = _FOOTNOTE_OPEN_RE.sub("", text)
    text = _FOOTNOTE_CLOSE_RE.sub("", text)
    text = _EMPTY_CELL_RE.sub("", text)
    text = _TABLE_TAG_RE.sub("", text)
    text = _IMAGE_TOKEN_RE.sub("", text)
    # Drop tag-only lines before collapsing the rest of the HTML tags.
    text = _EMPTY_TABLE_ROW_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)

    out_lines: list[str] = []
    for original_line in text.splitlines():
        stripped_line = original_line.rstrip()
        if not stripped_line.strip():
            continue
        if stripped_line.lstrip().startswith("★") and not _HAS_NUMERIC_FINGERPRINT.search(
            stripped_line
        ):
            # Template safety/torque reminder without any value → noise.
            continue
        if _NOISE_LINE_RE.match(stripped_line):
            continue
        # Deduplicate adjacent identical lines within a chapter as well,
        # collapse >1 blank to one.
        if out_lines and out_lines[-1] == stripped_line:
            continue
        out_lines.append(stripped_line)

    return "\n".join(out_lines)
