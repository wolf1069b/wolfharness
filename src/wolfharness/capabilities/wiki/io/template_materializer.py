"""Template-based entity materialization from source packet fields.

Assembles entity page content directly from structured packet body fields
(``source_subject``, ``explicit_facts``, ``parts_and_specs``,
``evidence_map``, ``ordered_actions``) without invoking an LLM.

This is the fast path for single-packet candidates where the packet already
carries all source-backed content needed for a draft entity page.  The output
is a full markdown page with YAML frontmatter, ready for ``write_entities_batch``.

Multi-packet merge candidates (where identity reconciliation is needed) stay on
the LLM worker path — ``plan_materialization_work`` marks them ``strategy=llm``.
"""

from __future__ import annotations

from typing import Any


__all__ = [
    "assemble_template_entity",
    "strip_device_prefix",
]

# ponytail: threshold for image filtering — above this count, captions are
# matched against object_name to drop irrelevant chapter-level images.
_IMAGE_FILTER_THRESHOLD = 10


def strip_device_prefix(object_name: str, device_id: str, series_id: str) -> str:
    """Strip a device model prefix from *object_name* if present.

    Uses the build's actual ``device_id`` / ``series_id`` (e.g. ``SY75C`` /
    ``SY75``) so the check is generic across manuals — no hardcoded prefix list.
    Returns the original *object_name* when no prefix matches or the stripped
    result would be empty.
    """
    for prefix in (device_id, series_id):
        if not prefix:
            continue
        if object_name.startswith(prefix):
            stripped = object_name[len(prefix) :].lstrip("-_/ :")
            if stripped:
                return stripped
    return object_name


def _yaml_escape(value: str) -> str:
    """Escape a string for safe YAML scalar usage."""
    # Wrap in quotes if it contains characters that would confuse YAML parsing.
    if any(
        ch in value
        for ch in (
            ":",
            "#",
            "'",
            '"',
            "\n",
            "{",
            "}",
            "[",
            "]",
            ",",
            "&",
            "*",
            "!",
            "|",
            ">",
            "%",
            "@",
            "`",
        )
    ):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    return value


def _concept_id_prefix(concept: str) -> str:
    """Return the frontmatter ``id`` prefix for a concept (e.g. Component → COMP)."""
    mapping = {
        "Component": "COMP",
        "DTC": "DTC",
        "Device": "DEV",
        "Symptom": "SYM",
        "Fault": "FLT",
        "Procedure": "PROC",
        "OP": "OP",
    }
    return mapping.get(concept, concept.upper()[:4])


def _format_parts_and_specs(parts_and_specs: Any) -> list[str]:
    """Render ``parts_and_specs`` dict as markdown bullet lines."""
    lines: list[str] = []
    if not isinstance(parts_and_specs, dict):
        return lines
    for key, value in parts_and_specs.items():
        if not str(key).strip():
            continue
        lines.append(f"- **{key}**: {value}")
    return lines


def _format_explicit_facts(explicit_facts: Any, limit: int = 5) -> list[str]:
    """Render ``explicit_facts`` list as markdown bullet lines (capped)."""
    lines: list[str] = []
    if not isinstance(explicit_facts, list):
        return lines
    for fact in explicit_facts[:limit]:
        if isinstance(fact, str) and fact.strip():
            lines.append(f"- {fact.strip()}")
    return lines


def _format_ordered_actions(ordered_actions: Any) -> list[str]:
    """Render ``ordered_actions`` list as numbered steps."""
    lines: list[str] = []
    if not isinstance(ordered_actions, list):
        return lines
    for idx, action in enumerate(ordered_actions, 1):
        if not isinstance(action, dict):
            continue
        action_name = str(action.get("action", action.get("step", ""))).strip()
        if not action_name:
            continue
        criterion = str(action.get("decision_criterion", "")).strip()
        if criterion:
            lines.append(f"{idx}. **{action_name}** — 判定标准: {criterion}")
        else:
            lines.append(f"{idx}. **{action_name}**")
    return lines


def _format_sources(evidence_map: Any, source_uris: list[str]) -> list[str]:
    """Deduce source URIs from evidence_map or fallback to source_uris."""
    uris: list[str] = []
    seen: set[str] = set()
    if isinstance(evidence_map, list):
        for entry in evidence_map:
            if not isinstance(entry, dict):
                continue
            uri = str(entry.get("source_uri", "")).strip()
            if uri and uri not in seen:
                uris.append(uri)
                seen.add(uri)
    for uri in source_uris:
        uri = str(uri).strip()
        if uri and uri not in seen:
            uris.append(uri)
            seen.add(uri)
    return uris


def _format_images(images: Any, object_name: str = "") -> list[str]:
    """Render ``images`` list as markdown image references.

    When *object_name* is provided and there are more than 10 images, filter
    to only include images whose caption shares at least one character with
    *object_name*.  This is a safety net for single-candidate packets that
    still carry many chapter-level images — only images whose caption relates
    to the entity survive.
    """
    if not isinstance(images, list):
        return []

    filtered = images
    if object_name and len(images) > _IMAGE_FILTER_THRESHOLD:
        name_chars = {ch for ch in object_name if not ch.isspace()}
        filtered = [
            img
            for img in images
            if isinstance(img, dict) and any(ch in str(img.get("caption", "")) for ch in name_chars)
        ]

    lines: list[str] = []
    for img in filtered:
        if not isinstance(img, dict):
            continue
        uri = str(img.get("uri", "")).strip()
        if not uri:
            continue
        caption = str(img.get("caption", "")).strip()
        lines.append(f"![{caption}]({uri})")
        lines.append("")
    return lines


def _component_body_sections(
    *,
    source_subject: str,
    working_mechanism: str,
    parts_and_specs: Any,
) -> list[str]:
    """Body sections for Component pages: identity + normal working principle.

    Component pages carry no parameter tables, thresholds, wiring info or
    fault propagation.  ``## 工作机理`` is always emitted so missing
    ``working_mechanism`` gaps stay visible.
    """
    lines: list[str] = []
    if source_subject:
        lines.append("## 总成概览")
        lines.append("")
        lines.append(source_subject)
        lines.append("")

    lines.append("## 工作机理")
    lines.append("")
    lines.append(
        working_mechanism.strip() or "> 工作机理待补充: 当前 packet 未提供 working_mechanism。"
    )
    lines.append("")

    spec_lines = _format_parts_and_specs(parts_and_specs)
    if spec_lines:
        lines.append("## 组成零件")
        lines.append("")
        lines.extend(spec_lines)
        lines.append("")
    return lines


def _fault_body_sections(
    *,
    source_subject: str,
    failure_mechanism: str,
) -> list[str]:
    """Body sections for Fault pages: failure description + mechanism.

    ``## 失效机理`` is always emitted so missing ``failure_mechanism``
    gaps stay visible.  Observable symptoms / 影响范围 are populated by
    relation workers, not the template path.
    """
    lines: list[str] = []
    if source_subject:
        lines.append("## 失效描述")
        lines.append("")
        lines.append(source_subject)
        lines.append("")

    lines.append("## 失效机理")
    lines.append("")
    lines.append(
        failure_mechanism.strip() or "> 失效机理待补充: 当前 packet 未提供 failure_mechanism。"
    )
    lines.append("")
    return lines


def assemble_template_entity(
    *,
    packet_body: dict[str, Any],
    concept: str,
    class_name: str,
    object_name: str,
    device_model: str,
    source_uris: list[str] | None = None,
) -> str:
    """Assemble a full entity page (frontmatter + body) from packet body fields.

    Only sections with content are emitted — no empty headings.  Component
    pages are the exception: ``## 工作机理`` is always emitted (with a
    placeholder when ``working_mechanism`` is missing) so content
    gaps stay visible.  Fault pages likewise always emit ``## 失效机理``.
    The output is a draft entity page (``status: draft``)
    ready for ``write_entities_batch``.

    Args:
        packet_body: The ``packet`` field from a stored source packet.
        concept: Entity concept (Component, DTC, Procedure, …).
        class_name: Logical assembly path or controller role.
        object_name: Brand+model or mechanical type name.
        device_model: Build ``device_id`` or ``series_id`` for applicable_models.
        source_uris: Fallback source URIs from the packet record.

    Returns:
        Full markdown content string with YAML frontmatter.
    """
    source_subject = str(packet_body.get("source_subject", "")).strip()
    parts_and_specs = packet_body.get("parts_and_specs")
    explicit_facts = packet_body.get("explicit_facts")
    ordered_actions = packet_body.get("ordered_actions")
    working_mechanism = str(packet_body.get("working_mechanism", ""))
    failure_mechanism = str(packet_body.get("failure_mechanism", ""))
    images = packet_body.get("images")

    # ── Frontmatter ──────────────────────────────────────────────────────
    id_prefix = _concept_id_prefix(concept)
    fm_lines = [
        "---",
        f"id: {id_prefix}-{object_name}",
        f"title: {object_name}",
        f"description: {_yaml_escape(source_subject)}",
        f"class_name: {class_name}",
        f"object_name: {object_name}",
        "status: draft",
    ]
    if device_model:
        fm_lines.append("applicable_models:")
        fm_lines.append(f"  - {device_model}")
    fm_lines.append("---")

    # ── Body sections (only non-empty) ───────────────────────────────────
    body_lines: list[str] = []

    if concept == "Component":
        body_lines.extend(
            _component_body_sections(
                source_subject=source_subject,
                working_mechanism=working_mechanism,
                parts_and_specs=parts_and_specs,
            )
        )
    elif concept == "Fault":
        body_lines.extend(
            _fault_body_sections(
                source_subject=source_subject,
                failure_mechanism=failure_mechanism,
            )
        )
    else:
        if source_subject:
            body_lines.append("## 总成概览")
            body_lines.append("")
            body_lines.append(source_subject)
            body_lines.append("")

        spec_lines = _format_parts_and_specs(parts_and_specs)
        if spec_lines:
            body_lines.append("## 规格参数")
            body_lines.append("")
            body_lines.extend(spec_lines)
            body_lines.append("")

        fact_lines = _format_explicit_facts(explicit_facts)
        if fact_lines:
            body_lines.append("## 关键事实")
            body_lines.append("")
            body_lines.extend(fact_lines)
            body_lines.append("")

        action_lines = _format_ordered_actions(ordered_actions)
        if action_lines:
            body_lines.append("## 步骤")
            body_lines.append("")
            body_lines.extend(action_lines)
            body_lines.append("")

    # ── Images (shared across all concepts) ──────────────────────────────
    image_lines = _format_images(images, object_name=object_name)
    if image_lines:
        body_lines.append("## 附图")
        body_lines.append("")
        body_lines.extend(image_lines)

    return "\n".join(fm_lines) + "\n" + "\n".join(body_lines)
