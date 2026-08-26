"""Re-ingestion skip and LLM-merge routing in ``materialize_template_batch``.

Regression scenario: a previous build materialised entities from source packets.
On re-ingestion (same manual, same wiki_root), the template batch must NOT
blindly overwrite existing draft entities that have real content.

Three behaviours tested:
- Content equivalence: identical body sections → skip (no-op, saves compute).
- Substantive content: existing draft with real text in any ``##`` section
  → route to ``fallback_to_llm`` so an LLM worker can merge.
- Empty template: existing draft with only placeholders → overwrite (correct).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wolfharness.capabilities.wiki.planning.materialization import (
    _has_substantive_content,
)
from wolfharness.capabilities.wiki.wiki_build_tools import WikiBuildTools


if TYPE_CHECKING:
    from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────


def _entity(
    sections: dict[str, str],
    *,
    title: str,
    status: str = "draft",
) -> str:
    """Render a minimal wiki entity (frontmatter + ``##`` sections)."""
    fm = "---\n"
    fm += f"title: {title}\n"
    fm += f"entity_status: {status}\n"
    fm += f"object_name: {title}\n"
    fm += "---\n\n"
    body = "\n\n".join(f"## {name}\n\n{text.strip()}" for name, text in sections.items())
    return fm + body + "\n"


def _make_packet(
    *,
    packet_id: str,
    build_id: str,
    doc_id: str,
    concept: str,
    class_name: str,
    object_name: str,
    working_mechanism: str = "主泵由齿轮驱动,液压油从油箱吸入后加压输出。",
    source_subject: str = "主泵工作原理",
) -> dict[str, object]:
    """Build a minimal source packet matching _materialization_candidates expectations."""
    return {
        "packet_id": packet_id,
        "build_id": build_id,
        "doc_id": doc_id,
        "status": "complete",
        "source_uris": [],
        "source_hash": "abc123",
        "extractor_config_hash": "cfg_v1",
        "packet": {
            "kind": "entity_extract",
            "source_subject": source_subject,
            "working_mechanism": working_mechanism,
            "explicit_facts": [],
            "parts_and_specs": [],
            "evidence_map": [],
            "ordered_actions": [],
            "normalized_identity_candidates": [
                {
                    "concept": concept,
                    "class_name": class_name,
                    "object_name": object_name,
                }
            ],
        },
    }


@pytest.fixture
def wiki_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WikiBuildTools:
    """A WikiBuildTools instance over a fresh local wiki root."""
    monkeypatch.setenv("WIKI_STORAGE_BACKEND", "local")
    wiki_root = tmp_path / "wiki"
    library_root = tmp_path / "library"
    wiki_root.mkdir()
    library_root.mkdir()
    tools = WikiBuildTools(wiki_root, library_root)
    # Minimal checkpoint so materialize_template_batch can find device info.
    tools.store.write_json(
        "index/build_checkpoint.json",
        {"build_id": "test_build", "doc_id": "SY75C", "device_id": "", "series_id": ""},
    )
    return tools


# ── Unit: _has_substantive_content ───────────────────────────────────────


@pytest.mark.unit
class TestHasSubstantiveContent:
    """Pure-function tests for the substantive-content detector."""

    def test_empty_sections_returns_false(self) -> None:
        content = _entity({}, title="X")
        assert _has_substantive_content(content) is False

    def test_placeholder_only_returns_false(self) -> None:
        content = _entity({"工作机理": "待补充"}, title="X")
        assert _has_substantive_content(content) is False

    def test_real_content_returns_true(self) -> None:
        content = _entity({"工作机理": "液压泵由发动机驱动运转"}, title="X")
        assert _has_substantive_content(content) is True

    def test_mixed_placeholder_and_real_returns_true(self) -> None:
        content = _entity(
            {"工作机理": "液压泵由发动机驱动运转", "失效机理": "待补充"},
            title="X",
        )
        assert _has_substantive_content(content) is True

    def test_non_core_section_real_returns_true(self) -> None:
        """Content in a non-core section (e.g. ## 技术参数) should count."""
        content = _entity(
            {"工作机理": "待补充", "技术参数": "额定压力: 34.3 MPa"},
            title="X",
        )
        assert _has_substantive_content(content) is True

    def test_frontmatter_only_returns_false(self) -> None:
        """An entity with frontmatter but no body sections has no substance."""
        content = "---\ntitle: X\nstatus: draft\n---\n"
        assert _has_substantive_content(content) is False


# ── Integration: re-ingestion routing in materialize_template_batch ──────


@pytest.mark.unit
class TestReingestRouting:
    """Template batch must not overwrite existing draft entities with content."""

    def test_identical_content_skipped(self, wiki_tools: WikiBuildTools) -> None:
        """Second call with same packet → content equivalence skip."""
        tools = wiki_tools
        packet = _make_packet(
            packet_id="pkt_001",
            build_id="test_build",
            doc_id="SY75C",
            concept="Component",
            class_name="液压系统",
            object_name="主泵",
        )
        tools.store.write_json("source_packets/pkt_001.json", packet)
        entity_uri = tools.store.entity_uri("Component", "液压系统", "主泵")

        # First call: creates the entity.
        result1 = tools.materialize_template_batch(
            packet_ids=["pkt_001"],
            entity_uris=[entity_uri],
            build_id="test_build",
            doc_id="SY75C",
        )
        assert result1["written_count"] == 1
        assert result1["skipped_existing"] == 0

        # Second call: identical content → skip.
        result2 = tools.materialize_template_batch(
            packet_ids=["pkt_001"],
            entity_uris=[entity_uri],
            build_id="test_build",
            doc_id="SY75C",
        )
        assert result2["written_count"] == 0
        assert result2["skipped_existing"] == 1

    def test_substantive_draft_routed_to_llm(self, wiki_tools: WikiBuildTools) -> None:
        """Existing draft with real content → fallback_to_llm, not overwrite."""
        tools = wiki_tools
        packet = _make_packet(
            packet_id="pkt_002",
            build_id="test_build",
            doc_id="SY75C",
            concept="Component",
            class_name="液压系统",
            object_name="多路阀",
        )
        tools.store.write_json("source_packets/pkt_002.json", packet)
        entity_uri = tools.store.entity_uri("Component", "液压系统", "多路阀")

        # Simulate a previous LLM worker that enriched a NON-core section
        # while leaving the core section (## 工作机理) as placeholder.
        # This is the scenario _has_substantive_content is designed to catch:
        # core_section check passes it through, but substantive content
        # in another section should prevent blind overwrite.
        enriched = _entity(
            {
                "工作机理": "待补充",
                "技术参数": "最大流量: 200 L/min, 额定压力: 31.5 MPa",
            },
            title="多路阀",
        )
        tools.store.write_entity("Component", "液压系统", "多路阀", enriched)

        result = tools.materialize_template_batch(
            packet_ids=["pkt_002"],
            entity_uris=[entity_uri],
            build_id="test_build",
            doc_id="SY75C",
        )
        assert result["written_count"] == 0
        assert result["skipped_existing"] == 0
        fallback_list: list[dict[str, str]] = result["fallback_to_llm"]  # type: ignore[assignment]
        assert len(fallback_list) >= 1
        fallback = fallback_list[0]
        assert fallback["entity_uri"] == entity_uri
        assert fallback["written"] is False
        assert "merge" in fallback["reason"]

    def test_empty_template_overwritten(self, wiki_tools: WikiBuildTools) -> None:
        """Existing draft with only placeholders → overwrite (no content lost)."""
        tools = wiki_tools
        packet = _make_packet(
            packet_id="pkt_003",
            build_id="test_build",
            doc_id="SY75C",
            concept="Component",
            class_name="液压系统",
            object_name="先导泵",
        )
        tools.store.write_json("source_packets/pkt_003.json", packet)
        entity_uri = tools.store.entity_uri("Component", "液压系统", "先导泵")

        # Existing entity is a pure placeholder template.
        placeholder = _entity({"工作机理": "待补充"}, title="先导泵")
        tools.store.write_entity("Component", "液压系统", "先导泵", placeholder)

        result = tools.materialize_template_batch(
            packet_ids=["pkt_003"],
            entity_uris=[entity_uri],
            build_id="test_build",
            doc_id="SY75C",
        )
        assert result["written_count"] == 1
        assert result["skipped_existing"] == 0

    def test_confirmed_entity_skipped(self, wiki_tools: WikiBuildTools) -> None:
        """Confirmed entity → always skipped regardless of content."""
        tools = wiki_tools
        packet = _make_packet(
            packet_id="pkt_004",
            build_id="test_build",
            doc_id="SY75C",
            concept="Component",
            class_name="液压系统",
            object_name="回转马达",
        )
        tools.store.write_json("source_packets/pkt_004.json", packet)
        entity_uri = tools.store.entity_uri("Component", "液压系统", "回转马达")

        confirmed = _entity(
            {"工作机理": "回转马达驱动上车回转"},
            title="回转马达",
            status="confirmed",
        )
        tools.store.write_entity("Component", "液压系统", "回转马达", confirmed)

        result = tools.materialize_template_batch(
            packet_ids=["pkt_004"],
            entity_uris=[entity_uri],
            build_id="test_build",
            doc_id="SY75C",
        )
        assert result["written_count"] == 0
        assert result["skipped_existing"] == 1

    def test_dtc_with_content_routed_to_llm(self, wiki_tools: WikiBuildTools) -> None:
        """DTC entities (core_section=None) with content → fallback_to_llm.

        Previously, DTC/Procedure/Symptom were ALWAYS overwritten because
        core_section was None.  Now _has_substantive_content catches them.
        """
        tools = wiki_tools
        packet = _make_packet(
            packet_id="pkt_dtc_001",
            build_id="test_build",
            doc_id="SY75C",
            concept="DTC",
            class_name="发动机",
            object_name="E01",
            working_mechanism="",  # DTC doesn't use working_mechanism
            source_subject="故障码E01: 发动机水温过高",
        )
        # DTC packet body needs different fields than Component
        dtc_body = packet["packet"]  # type: ignore[index]
        dtc_body["kind"] = "entity_extract"  # type: ignore[union-attr]
        dtc_body.pop("working_mechanism", None)  # type: ignore[union-attr]
        tools.store.write_json("source_packets/pkt_dtc_001.json", packet)
        entity_uri = tools.store.entity_uri("DTC", "发动机", "E01")

        # Simulate existing DTC with real content from previous LLM worker.
        existing_dtc = _entity(
            {"故障描述": "发动机冷却液温度超过阈值110°C", "可能原因": "节温器故障或冷却液不足"},
            title="E01",
        )
        tools.store.write_entity("DTC", "发动机", "E01", existing_dtc)

        result = tools.materialize_template_batch(
            packet_ids=["pkt_dtc_001"],
            entity_uris=[entity_uri],
            build_id="test_build",
            doc_id="SY75C",
        )
        # Should NOT overwrite — should route to LLM merge.
        assert result["written_count"] == 0
        fallback_list: list[dict[str, str]] = result["fallback_to_llm"]  # type: ignore[assignment]
        assert len(fallback_list) >= 1
        fallback = fallback_list[0]
        assert fallback["entity_uri"] == entity_uri
        assert fallback["written"] is False
