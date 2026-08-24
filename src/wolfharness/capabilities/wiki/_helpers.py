"""Module-level helpers and constants extracted from mcp_server.py."""

from __future__ import annotations

from hashlib import sha256
import json
import logging
import os
import re
from typing import TYPE_CHECKING

from wolfharness.capabilities.wiki.quality import (
    parse_frontmatter,
)
from wolfharness.capabilities.wiki.schema_loader import get_schema_version
from wolfharness.capabilities.wiki.validation import (
    ENTITY_VALIDATION_HOOKS,
    FORMAL_WRITE_EXCLUDED_HOOKS,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


_FORMAL_WRITE_HOOKS = tuple(
    hook for hook in ENTITY_VALIDATION_HOOKS if hook.name not in FORMAL_WRITE_EXCLUDED_HOOKS
)
_RELATION_CLOSURE_READY_STAGES = frozenset(
    {
        "materialized",
        "relation_closure",
        "audit_fix",
        "auditing",
        "audited",
        "validate",
        "publishing",
        "finalized",
        "done",
        "recovered",
    },
)


def _entity_batch_limit() -> int:
    """Return the configured bounded entity batch size."""
    try:
        return max(1, min(100, int(os.environ.get("WIKI_ENTITY_BATCH_SIZE", "20"))))
    except ValueError:
        return 20


def _materialization_task_byte_limit() -> int:
    """Return the configurable upper bound for one embedded 1B task."""
    try:
        return max(
            16_384,
            min(524_288, int(os.environ.get("WIKI_MATERIALIZATION_TASK_BYTES", "98304"))),
        )
    except ValueError:
        return 98_304


def _io_worker_limit() -> int:
    """Return the bounded parallel I/O width for one build service."""
    try:
        return max(1, min(64, int(os.environ.get("WIKI_IO_WORKERS", "16"))))
    except ValueError:
        return 16


def _natural_path_key(path: str) -> tuple[tuple[int, int | str], ...]:
    """Sort chapter paths by numeric prefixes instead of dictionary order."""
    key: list[tuple[int, int | str]] = []
    for part in re.split(r"(\d+)", path.casefold()):
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part))
    return tuple(key)


_CHAPTER_COMPONENT_RE = re.compile(r"^\d+(?:[._]\d+)*")
_BOM_ENRICH_PLACEHOLDER_MARKERS = (
    "工作机理由 bom_enrich worker",
    "BOM 仅提供身份",
    "由维修手册章节抽取阶段补充",
    "工作机理待补充",
)
_BOM_PATH_PLACEHOLDER_RE = re.compile(r"(?:\.\.\.|…|\b(?:todo|tbd|unknown)\b)", re.IGNORECASE)


def _chapter_component_name(parts: Sequence[str], fallback: str) -> str:
    """Return the nearest numeric chapter directory for artifact-backed leaves."""
    for part in reversed(parts):
        if _CHAPTER_COMPONENT_RE.match(part):
            return part
    return fallback


def _chapter_idempotency_key(build_id: str, doc_id: str, uri: str) -> str:
    """Stable 12-hex idempotency key shared by all chapter dispatch tools.

    ``preflight_build`` and ``browse_chapters`` must derive the same key
    for the same (build_id, doc_id, uri) so retries/restarts reuse already
    recorded source packets whichever tool produced them.  Twelve hex chars
    (48 bits) keeps keys short for task descriptions; the per-build birthday
    collision probability at chapter scale (~4e-11 for 152 chapters) is
    negligible and a collision only affects in-build dedup, never integrity.
    """
    return sha256(
        f"{build_id}\x1f{doc_id}\x1f{uri}\x1f{get_schema_version()}".encode(),
    ).hexdigest()[:12]


def _with_publication_state(content: str, *, deprecated: bool = False) -> str:
    """Stamp machine publication state without implying human approval."""
    frontmatter = parse_frontmatter(content)
    conflict_pending = str(frontmatter.get("conflict_pending", "")).strip().lower() in {
        "true",
        "yes",
        "1",
    }
    publication = "deprecated" if deprecated else "published"
    updates = {
        "publication_state": "blocked" if conflict_pending else publication,
        "validation_state": "machine_validated",
        "review_state": "unreviewed",
        "status": "deprecated" if deprecated else ("draft" if conflict_pending else "confirmed"),
    }
    lines = content.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        body = content if content.endswith("\n") else content + "\n"
        header = "---\n" + "".join(f"{key}: {value}\n" for key, value in updates.items()) + "---\n"
        return header + body
    end = next((index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"), None)
    if end is None:
        return content
    seen: set[str] = set()
    for index in range(1, end):
        line = lines[index]
        if line[:1].isspace():
            continue
        field = re.match(r"^([A-Za-z_]\w*)\s*:", line)
        if field is None or field.group(1) not in updates:
            continue
        key = field.group(1)
        lines[index] = f"{key}: {updates[key]}\n"
        seen.add(key)
    missing = [f"{key}: {value}\n" for key, value in updates.items() if key not in seen]
    lines[end:end] = missing
    return "".join(lines)


_CONFLICT_IGNORED_FRONTMATTER = frozenset(
    {
        "id",
        "title",
        "description",
        "class_name",
        "object_name",
        "status",
        "sources",
        "applicable_models",
        "aliases",
        "publication_state",
        "validation_state",
        "review_state",
        "conflict_pending",
        "conflict_refs",
    },
)

_PARAMETER_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>kpa|mpa|pa|bar|kv|mv|v|ka|ma|a|khz|hz|mm|cm|m|℃|°c|c)?(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_NON_FACT_LABEL_RE = re.compile(
    r"^(?:第|章节?|chapter|section|页|page|图|表|编号|序号|步骤|step|no)\s*$",
    re.IGNORECASE,
)
_UNIT_SCALE: dict[str, tuple[str, float]] = {
    "pa": ("pressure_pa", 1.0),
    "kpa": ("pressure_pa", 1_000.0),
    "mpa": ("pressure_pa", 1_000_000.0),
    "bar": ("pressure_pa", 100_000.0),
    "v": ("voltage_v", 1.0),
    "mv": ("voltage_v", 0.001),
    "kv": ("voltage_v", 1_000.0),
    "a": ("current_a", 1.0),
    "ma": ("current_a", 0.001),
    "ka": ("current_a", 1_000.0),
    "hz": ("frequency_hz", 1.0),
    "khz": ("frequency_hz", 1_000.0),
    "mm": ("length_mm", 1.0),
    "cm": ("length_mm", 10.0),
    "m": ("length_mm", 1_000.0),
    "℃": ("temperature_c", 1.0),
    "°c": ("temperature_c", 1.0),
    "c": ("temperature_c", 1.0),
}

_DIMENSION_CN: dict[str, str] = {
    "voltage_v": "电压",
    "pressure_pa": "压力",
    "current_a": "电流",
    "frequency_hz": "频率",
    "length_mm": "长度",
    "temperature_c": "温度",
    "unitless": "数值",
}


def _humanize_fact_key(key: str) -> str:
    """Render a machine fact key as a human-readable Chinese label."""
    if key.startswith("parameter:"):
        _, label, dimension = key.split(":", 2)
        return f"参数「{label}」的{_DIMENSION_CN.get(dimension, '数值')}"
    if key.startswith("fm:"):
        return f"frontmatter 字段「{key.split(':', 1)[1]}」"
    if key.startswith("body-state:"):
        return f"状态描述「{key.split(':', 1)[1]}」"
    return key


def _parameter_fact_map(label: str, value: object) -> dict[str, set[str]]:
    """Return normalized parameter values keyed by their semantic label."""
    text = str(value)
    matches = list(_PARAMETER_RE.finditer(text))
    if not matches:
        return {}
    label_without_values = _PARAMETER_RE.sub(" ", label)
    normalized_label = (
        re
        .sub(r"\s+", " ", re.sub(r"[^\w\u4e00-\u9fff]+", " ", label_without_values))
        .strip()
        .casefold()
    )
    if not normalized_label or _NON_FACT_LABEL_RE.fullmatch(normalized_label):
        return {}
    facts: dict[str, set[str]] = {}
    for match in matches:
        number = float(match.group("value"))
        unit = (match.group("unit") or "").casefold()
        dimension, scale = _UNIT_SCALE.get(unit, ("unitless", 1.0))
        normalized_value = round(number * scale, 9)
        facts.setdefault(f"parameter:{normalized_label}:{dimension}", set()).add(
            str(normalized_value)
        )
    return facts


def _conflict_fact_map(content: str) -> dict[str, set[str]]:
    """Extract comparable facts without treating additive content as conflict."""
    frontmatter = parse_frontmatter(content)
    facts: dict[str, set[str]] = {}
    for key, value in frontmatter.items():
        if key in _CONFLICT_IGNORED_FRONTMATTER or value in (None, "", [], {}):
            continue
        normalized = _parameter_fact_map(key, value)
        if normalized:
            for fact_key, fact_values in normalized.items():
                facts.setdefault(fact_key, set()).update(fact_values)
        elif isinstance(value, (str, int, float, bool)):
            facts[f"fm:{key}"] = {json.dumps(value, ensure_ascii=False, sort_keys=True)}

    lines = content.splitlines()
    in_frontmatter = bool(lines and lines[0].strip() == "---")
    opposite_words = (
        ("正常", "异常"),
        ("有", "无"),
        ("通", "断"),
        ("高", "低"),
        ("开", "关"),
        ("允许", "禁止"),
        ("支持", "不支持"),
        ("准确", "不准"),
    )
    for line in lines[1:] if in_frontmatter else lines:
        if in_frontmatter and line.strip() == "---":
            in_frontmatter = False
            continue
        if in_frontmatter:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "viking://" in stripped:
            continue
        parameter_matches = list(_PARAMETER_RE.finditer(stripped))
        has_explicit_unit = any(match.group("unit") for match in parameter_matches)
        if (
            parameter_matches
            and not has_explicit_unit
            and re.match(r"^\s*(?:\d+[.)、]|第\s*\d+\s*[章节])", stripped, re.IGNORECASE)
        ):
            # Chapter/step numbering is navigation metadata, not a domain
            # fact. It must never become an OPA numeric conflict.
            continue
        normalized = _parameter_fact_map(stripped, stripped)
        if normalized:
            for fact_key, fact_values in normalized.items():
                facts.setdefault(fact_key, set()).update(fact_values)
        else:
            compact = re.sub(r"\s+", " ", stripped)
            for first, second in opposite_words:
                if (first in compact) == (second in compact):
                    continue
                base = compact.replace(first, "").replace(second, "").strip()
                facts.setdefault(f"body-state:{base}", set()).add(
                    first if first in compact else second
                )
    return facts


def _conflicting_facts(current: str, candidate: str) -> set[str]:
    """Return only facts whose existing and candidate values are incompatible."""
    current_facts = _conflict_fact_map(current)
    candidate_facts = _conflict_fact_map(candidate)
    conflicts: set[str] = set()
    for key in current_facts.keys() & candidate_facts.keys():
        current_values = current_facts[key]
        candidate_values = candidate_facts[key]
        if current_values.isdisjoint(candidate_values):
            humanized_key = _humanize_fact_key(key)
            current_vals = ", ".join(sorted(current_values))
            candidate_vals = ", ".join(sorted(candidate_values))
            conflicts.add(f"{humanized_key}：已有内容为 {current_vals}，新内容为 {candidate_vals}")
    return conflicts


def _internal_conflicting_facts(content: str) -> set[str]:
    """Find incompatible parameter values repeated inside one candidate.

    Body lines are grouped by ``##``/``###`` headings so the same parameter
    in different subsections (for example distinct measurement points on one
    page) is not reported as a conflict.  Without section headings the whole
    document stays a single comparison group, matching the previous
    whole-page heuristic.
    """
    facts: dict[str, set[str]] = {}

    def add_value(bucket: dict[str, set[str]], label: str, value: str) -> None:
        if re.search(r"\d\s*(?:-|~|～|至|到)\s*-?\d", value):
            return
        for key, values in _parameter_fact_map(label, value).items():
            bucket.setdefault(key, set()).update(values)

    lines = content.splitlines()
    in_frontmatter = bool(lines and lines[0].strip() == "---")
    body_lines = lines[1:] if in_frontmatter else lines
    has_sections = any(re.match(r"^#{2,3}\s", line.strip()) for line in body_lines)
    if not has_sections:
        frontmatter = parse_frontmatter(content)
        for key, value in frontmatter.items():
            if isinstance(value, str):
                add_value(facts, key, value)
    buckets: list[dict[str, set[str]]] = [facts]
    current: dict[str, set[str]] = facts
    for line in body_lines:
        if in_frontmatter and line.strip() == "---":
            in_frontmatter = False
            continue
        if in_frontmatter:
            continue
        stripped = line.strip()
        if re.match(r"^#{2,3}\s", stripped):
            current = {}
            buckets.append(current)
            continue
        if not stripped or stripped.startswith("#") or "://" in stripped:
            continue
        match = _PARAMETER_RE.search(stripped)
        if match:
            add_value(current, stripped[: match.start()], stripped)

    return {
        f"{_humanize_fact_key(key)}在同一文档内不一致：{', '.join(sorted(values))}"
        for bucket in buckets
        for key, values in bucket.items()
        if len(values) > 1
    }


if __name__ == "__main__":
    # Self-check: conflict strings must be human-readable Chinese, not machine keys.
    expected = "参数「传感器输出电压异常」的电压"
    assert _humanize_fact_key("parameter:传感器输出电压异常:voltage_v") == expected
    assert _humanize_fact_key("fm:title") == "frontmatter 字段「title」"
    assert _humanize_fact_key("body-state:压力正常") == "状态描述「压力正常」"
    assert _humanize_fact_key("unknown:key") == "unknown:key"
    cross = _conflicting_facts("电压 0.3v\n", "电压 4.7v\n")
    assert cross == {"参数「电压」的电压：已有内容为 0.3，新内容为 4.7"}, cross
    internal = _internal_conflicting_facts("电压 0.3v\n电压 4.7v\n")
    assert internal == {"参数「电压」的电压在同一文档内不一致：0.3, 4.7"}, internal
    print("humanized conflict format self-check passed")
