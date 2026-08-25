"""Expert-section preservation across pipeline writes and relation sync.

Regression scenario: an expert applies an OPL (either a single named section
or a whole-page external submission), then a later pipeline re-ingestion
writes the same entity.  The expert-owned section content must survive
verbatim while every other section receives the pipeline's content.

Covers:
- ``_preserve_expert_sections`` merge semantics (named section, removed
  section, full-entity authority, stale authority, no authority, errors).
- ``write_entity`` wiring: pipeline re-ingest after an expert edit.
- Relation-sync wiring: Component narrative links and Device diagnostic
  tables no longer overwrite expert-owned sections.
"""

from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING

import pytest

from wolfharness.capabilities.wiki.models import OPSModel
from wolfharness.capabilities.wiki.quality import extract_sections
from wolfharness.capabilities.wiki.wiki_build_tools import WikiBuildTools


if TYPE_CHECKING:
    from pathlib import Path


AUTHORITY_SECTION = "常见故障及故障机理"
PIPELINE_SECTION = "诊断流程"


def make_entity(
    sections: dict[str, str],
    *,
    title: str,
    status: str = "draft",
    frontmatter: dict[str, object] | None = None,
) -> str:
    """Render a minimal wiki entity (frontmatter + ``##`` sections)."""
    fields: dict[str, object] = {"title": title, "entity_status": status}
    if frontmatter:
        fields.update(frontmatter)
    lines: list[str] = ["---"]
    for key, value in fields.items():
        if isinstance(value, list):
            rendered = ", ".join(str(item) for item in value)
            lines.append(f"{key}: [{rendered}]")
        elif isinstance(value, bool):
            lines.append(f"{key}: {'true' if value else 'false'}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    body = "\n\n".join(f"## {name}\n\n{text.strip()}" for name, text in sections.items())
    return "\n".join(lines) + "\n\n" + body + "\n"


@pytest.fixture
def wiki_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WikiBuildTools:
    """A WikiBuildTools instance over a fresh local wiki root."""
    monkeypatch.setenv("WIKI_STORAGE_BACKEND", "local")
    wiki_root = tmp_path / "wiki"
    library_root = tmp_path / "library"
    wiki_root.mkdir()
    library_root.mkdir()
    return WikiBuildTools(wiki_root, library_root)


def authority(
    section: str, *, target_uri: str = "viking://resources/test/Entity"
) -> list[dict[str, str]]:
    """Shape of one ``get_expert_authority`` record (as returned post-OPL-apply)."""
    return [{"target_section": section, "target_uri": target_uri, "source": "opl"}]


def stub_authority(records: list[dict[str, str]]):
    """Return a ``get_expert_authority`` stand-in returning *records*."""

    def _get(*, target_uri: str = "", limit: int = 50) -> list[dict[str, str]]:
        return records

    return _get


@pytest.mark.unit
class TestPreserveExpertSections:
    """Unit semantics of the merge helper itself."""

    def test_named_section_kept_verbatim_others_updated(self, wiki_tools: WikiBuildTools) -> None:
        tools = wiki_tools
        tools.get_expert_authority = stub_authority(authority(AUTHORITY_SECTION))
        current = make_entity(
            {AUTHORITY_SECTION: "专家修正:低温无法启动", PIPELINE_SECTION: "旧流程"}, title="T"
        )
        candidate = make_entity(
            {AUTHORITY_SECTION: "管道覆盖", PIPELINE_SECTION: "新流程"}, title="T"
        )
        merged = tools._preserve_expert_sections(
            target_uri="u", current=current, candidate=candidate
        )
        sections = extract_sections(merged)
        assert sections[AUTHORITY_SECTION] == "专家修正:低温无法启动"
        assert sections[PIPELINE_SECTION] == "新流程"

    def test_removed_expert_section_is_restored(self, wiki_tools: WikiBuildTools) -> None:
        tools = wiki_tools
        tools.get_expert_authority = stub_authority(authority(AUTHORITY_SECTION))
        current = make_entity({AUTHORITY_SECTION: "专家内容", "其他": "保持"}, title="T")
        candidate = make_entity({"其他": "新内容"}, title="T")  # authority section missing
        merged = tools._preserve_expert_sections(
            target_uri="u", current=current, candidate=candidate
        )
        sections = extract_sections(merged)
        assert sections[AUTHORITY_SECTION] == "专家内容"
        assert sections["其他"] == "新内容"

    @pytest.mark.parametrize("phantom", ["external_opl", ""])
    def test_full_entity_authority_freezes_existing_sections(
        self, wiki_tools: WikiBuildTools, phantom: str
    ) -> None:
        """Empty or phantom target_section claims every existing section."""
        tools = wiki_tools
        tools.get_expert_authority = stub_authority(authority(phantom))
        current = make_entity({"A": "1", "B": "2"}, title="T")
        candidate = make_entity({"A": "X", "B": "Y", "C": "新建"}, title="T")
        merged = tools._preserve_expert_sections(
            target_uri="u", current=current, candidate=candidate
        )
        sections = extract_sections(merged)
        assert sections["A"] == "1"
        assert sections["B"] == "2"
        assert sections["C"] == "新建"

    def test_stale_section_authority_is_ignored(self, wiki_tools: WikiBuildTools) -> None:
        tools = wiki_tools
        tools.get_expert_authority = stub_authority(authority("已删除的章节"))
        current = make_entity({"A": "1"}, title="T")
        candidate = make_entity({"A": "新", "B": "2"}, title="T")
        merged = tools._preserve_expert_sections(
            target_uri="u", current=current, candidate=candidate
        )
        assert merged == candidate

    def test_no_authority_returns_candidate_unchanged(self, wiki_tools: WikiBuildTools) -> None:
        tools = wiki_tools
        tools.get_expert_authority = stub_authority([])
        current = make_entity({AUTHORITY_SECTION: "旧"}, title="T")
        candidate = make_entity({AUTHORITY_SECTION: "新"}, title="T")
        merged = tools._preserve_expert_sections(
            target_uri="u", current=current, candidate=candidate
        )
        assert merged == candidate

    def test_authority_lookup_error_returns_candidate(self, wiki_tools: WikiBuildTools) -> None:
        tools = wiki_tools

        def _boom(*, target_uri: str = "", limit: int = 50) -> list[dict[str, str]]:
            raise OSError("expert store unavailable")

        tools.get_expert_authority = _boom
        current = make_entity({AUTHORITY_SECTION: "旧"}, title="T")
        candidate = make_entity({AUTHORITY_SECTION: "新"}, title="T")
        merged = tools._preserve_expert_sections(
            target_uri="u", current=current, candidate=candidate
        )
        assert merged == candidate

    @pytest.mark.parametrize("phantom", ["external_opl", ""])
    def test_full_entity_authority_restores_dropped_frontmatter(
        self, wiki_tools: WikiBuildTools, phantom: str
    ) -> None:
        """Full-entity authority re-inserts expert frontmatter keys the candidate drops."""
        tools = wiki_tools
        tools.get_expert_authority = stub_authority(authority(phantom))
        current = make_entity(
            {"A": "1"},
            title="T",
            frontmatter={"experts_note": "专家纠错:油品标号", "affected_components": ["主泵"]},
        )
        candidate = make_entity(
            {"A": "X", "B": "新建"},
            title="T",
            frontmatter={"affected_components": ["新部件"]},
        )
        merged = tools._preserve_expert_sections(
            target_uri="u", current=current, candidate=candidate
        )
        # Dropped expert key is restored, candidate's non-empty renewal is kept.
        assert "专家纠错:油品标号" in merged
        assert "新部件" in merged
        sections = extract_sections(merged)
        assert sections["A"] == "1"
        assert sections["B"] == "新建"

    def test_full_entity_authority_keeps_candidate_relation_fields(
        self, wiki_tools: WikiBuildTools
    ) -> None:
        """Candidate non-empty frontmatter wins (pipeline relation renewals land)."""
        tools = wiki_tools
        tools.get_expert_authority = stub_authority(authority("external_opl"))
        current = make_entity(
            {"A": "1"}, title="T", frontmatter={"affected_components": ["旧件"]}
        )
        candidate = make_entity(
            {"A": "X"}, title="T", frontmatter={"affected_components": ["新件"]}
        )
        merged = tools._preserve_expert_sections(
            target_uri="u", current=current, candidate=candidate
        )
        assert "新件" in merged
        assert "旧件" not in merged

    def test_named_section_authority_does_not_touch_frontmatter(
        self, wiki_tools: WikiBuildTools
    ) -> None:
        """Section-scoped authority owns sections only, never the frontmatter."""
        tools = wiki_tools
        tools.get_expert_authority = stub_authority(authority(AUTHORITY_SECTION))
        current = make_entity(
            {AUTHORITY_SECTION: "专家内容", "B": "2"},
            title="T",
            frontmatter={"experts_note": "纠错"},
        )
        candidate = make_entity({AUTHORITY_SECTION: "新", "B": "2"}, title="T")
        merged = tools._preserve_expert_sections(
            target_uri="u", current=current, candidate=candidate
        )
        sections = extract_sections(merged)
        assert sections[AUTHORITY_SECTION] == "专家内容"
        assert "experts_note" not in merged
        assert "纠错" not in merged


@pytest.mark.integration
class TestPipelineWiring:
    """Expert edits survive a real re-ingestion through the write paths."""

    def test_write_entity_keeps_expert_section_after_reingest(
        self, wiki_tools: WikiBuildTools
    ) -> None:
        tools = wiki_tools
        store = tools.store
        concept, object_name = "Device", "SY75C"
        uri = store.entity_uri(concept, None, object_name)
        # Round 1: extraction writes the page.
        store.write_entity(
            concept, None, object_name,
            make_entity({AUTHORITY_SECTION: "旧表", PIPELINE_SECTION: "旧流程"}, title="SY75C"),
        )
        # Round 2: expert OPL applied → section content changed.
        expert = make_entity(
            {AUTHORITY_SECTION: "专家修正表", PIPELINE_SECTION: "旧流程"}, title="SY75C"
        )
        store.write_entity(concept, None, object_name, expert)
        tools.get_expert_authority = stub_authority(authority(AUTHORITY_SECTION, target_uri=uri))
        # Round 3: new manual ingestion rewrites the page with a fresh diff.
        current = store.read_entity(concept, None, object_name)
        assert current is not None
        expected_sha = sha256(current.encode("utf-8")).hexdigest()
        candidate = make_entity(
            {AUTHORITY_SECTION: "管道覆盖", PIPELINE_SECTION: "新流程"}, title="SY75C"
        )
        tools.write_entity(concept, "", object_name, candidate, expected_sha256=expected_sha)
        stored = store.read_entity(concept, None, object_name) or ""
        sections = extract_sections(stored)
        assert sections[AUTHORITY_SECTION] == "专家修正表"
        assert sections[PIPELINE_SECTION] == "新流程"

    def test_component_narrative_sync_preserves_expert_section(
        self, wiki_tools: WikiBuildTools
    ) -> None:
        tools = wiki_tools
        store = tools.store
        concept, clz, object_name = "Component", "关重件", "主泵"
        uri = store.entity_uri(concept, clz, object_name)
        # The canonical sync sections on Component pages (section_constants.py):
        # 常见失效模式 (fault links) and 拆装步骤 (procedure links).
        section, other = "常见失效模式", "拆装步骤"
        store.write_entity(
            concept, clz, object_name,
            make_entity({section: "老化磨损", other: "1. 断电"}, title="主泵"),
        )
        expert_body = "专家补充:叶片磨损致效率下降"
        expert = make_entity({section: expert_body, other: "1. 断电"}, title="主泵")
        store.write_entity(concept, clz, object_name, expert)
        tools.get_expert_authority = stub_authority(authority(section, target_uri=uri))
        # A linked Fault/Procedure write triggers the Component narrative sync.
        tools._sync_component_narrative_links_locked(
            uri,
            fault_links=[("主泵异响", "#fault-1")],
            procedure_links=[("拆卸主泵", "#proc-1")],
        )
        stored = store.read_entity(concept, clz, object_name) or ""
        sections = extract_sections(stored)
        # Expert-owned section is frozen: the new fault link did not land.
        assert sections[section] == expert_body
        assert "主泵异响" not in sections[section]
        # Non-expert section still receives sync content.
        assert "拆卸主泵" in sections[other]

    def test_device_diagnostic_sync_preserves_expert_sections(
        self, wiki_tools: WikiBuildTools
    ) -> None:
        tools = wiki_tools
        store = tools.store
        device_uri = store.entity_uri("Device", None, "SY75C")
        symptom_uri = store.entity_uri("Symptom", None, "无法启动")
        store.write_entity(
            "Symptom", None, "无法启动",
            make_entity({"诊断要点": "查电瓶"}, title="无法启动"),
        )
        store.write_entity(
            "Device", None, "SY75C",
            make_entity(
                {AUTHORITY_SECTION: "旧表", "控制器与故障码": "旧码"},
                title="SY75C",
                frontmatter={"symptom_refs": [symptom_uri]},
            ),
        )
        expert = make_entity(
            {AUTHORITY_SECTION: "专家诊断表", "控制器与故障码": "专家码表"},
            title="SY75C",
            frontmatter={"symptom_refs": [symptom_uri]},
        )
        store.write_entity("Device", None, "SY75C", expert)
        # Whole-page authority (external expert submission).
        tools.get_expert_authority = stub_authority(
            authority("external_opl", target_uri=device_uri)
        )
        tools._sync_device_diagnostic_links()
        stored = store.read_entity("Device", None, "SY75C") or ""
        sections = extract_sections(stored)
        # Expert sections are frozen even though the sync regenerated the
        # symptom table (frontmatter unaffected by the sync).
        assert sections[AUTHORITY_SECTION] == "专家诊断表"
        assert sections["控制器与故障码"] == "专家码表"
        assert "无法启动" not in sections[AUTHORITY_SECTION]

    def test_write_entities_batch_keeps_expert_section_after_reingest(
        self, wiki_tools: WikiBuildTools
    ) -> None:
        """Batch materialization must not overwrite expert-owned sections."""
        tools = wiki_tools
        store = tools.store
        concept, object_name = "Device", "SY75C"
        uri = store.entity_uri(concept, None, object_name)
        store.write_entity(
            concept, None, object_name,
            make_entity({AUTHORITY_SECTION: "旧表", PIPELINE_SECTION: "旧流程"}, title="SY75C"),
        )
        expert = make_entity(
            {AUTHORITY_SECTION: "专家修正表", PIPELINE_SECTION: "旧流程"}, title="SY75C"
        )
        store.write_entity(concept, None, object_name, expert)
        tools.get_expert_authority = stub_authority(authority(AUTHORITY_SECTION, target_uri=uri))
        current = store.read_entity(concept, None, object_name)
        assert current is not None
        expected_sha = sha256(current.encode("utf-8")).hexdigest()
        candidate = make_entity(
            {AUTHORITY_SECTION: "管道覆盖", PIPELINE_SECTION: "新流程"}, title="SY75C"
        )
        result = tools.write_entities_batch(
            [
                {
                    "concept": concept,
                    "class_name": "",
                    "object_name": object_name,
                    "content": candidate,
                    "expected_sha256": expected_sha,
                }
            ]
        )
        assert uri in result["uris"]
        stored = store.read_entity(concept, None, object_name) or ""
        sections = extract_sections(stored)
        assert sections[AUTHORITY_SECTION] == "专家修正表"
        assert sections[PIPELINE_SECTION] == "新流程"

    def test_apply_ops_keeps_expert_candidate(self, wiki_tools: WikiBuildTools) -> None:
        """Confirmed OPS apply must not be reverted by its own authority claim.

        Regression: ``apply_ops`` merged with conflict_policy="detect", so the
        record's own confirmed (but not yet applied) authority froze the page
        back to its pre-apply state and the expert change vanished silently.
        """
        tools = wiki_tools
        store = tools.store
        concept, object_name = "Device", "SY75C"
        uri = store.entity_uri(concept, None, object_name)
        store.write_entity(
            concept, None, object_name,
            make_entity({AUTHORITY_SECTION: "原始表", PIPELINE_SECTION: "原始流程"}, title="SY75C"),
        )
        current = store.read_entity(concept, None, object_name)
        assert current is not None
        expected_sha = sha256(current.encode("utf-8")).hexdigest()
        expert_candidate = make_entity(
            {AUTHORITY_SECTION: "专家修正表", PIPELINE_SECTION: "原始流程"}, title="SY75C"
        )
        tools._write_ops_model(
            OPSModel(
                ops_id="ops-expert-1",
                parent_opa="opa-expert-1",
                title="专家修正",
                status="confirmed",
                target_uri=uri,
                solution="修正故障机理表",
                analysis="原表缺少关键失效模式",
                retrieval_query="expert correction",
                candidate_content=expert_candidate,
                expected_sha256=expected_sha,
                reviewed_by="expert@test",
            )
        )
        result = tools.apply_ops("ops-expert-1")
        assert result["apply_status"] == "applied"
        stored = store.read_entity(concept, None, object_name) or ""
        sections = extract_sections(stored)
        assert sections[AUTHORITY_SECTION] == "专家修正表"
        assert "专家修正表" in stored

    def test_write_entity_external_authority_lands_candidate(
        self, wiki_tools: WikiBuildTools
    ) -> None:
        """The expert apply path bypasses preservation (contract for apply_ops)."""
        tools = wiki_tools
        store = tools.store
        concept, object_name = "Device", "SY75C"
        uri = store.entity_uri(concept, None, object_name)
        store.write_entity(
            concept, None, object_name,
            make_entity({AUTHORITY_SECTION: "旧表", PIPELINE_SECTION: "旧流程"}, title="SY75C"),
        )
        store.write_entity(
            concept, None, object_name,
            make_entity({AUTHORITY_SECTION: "专家内容", PIPELINE_SECTION: "旧流程"}, title="SY75C"),
        )
        tools.get_expert_authority = stub_authority(authority(AUTHORITY_SECTION, target_uri=uri))
        current = store.read_entity(concept, None, object_name)
        assert current is not None
        expected_sha = sha256(current.encode("utf-8")).hexdigest()
        candidate = make_entity(
            {AUTHORITY_SECTION: "专家再次修改", PIPELINE_SECTION: "新流程"}, title="SY75C"
        )
        tools.write_entity(
            concept,
            "",
            object_name,
            candidate,
            expected_sha256=expected_sha,
            conflict_policy="external_authority",
        )
        stored = store.read_entity(concept, None, object_name) or ""
        sections = extract_sections(stored)
        assert sections[AUTHORITY_SECTION] == "专家再次修改"
        assert sections[PIPELINE_SECTION] == "新流程"
