"""Timestamp-based ID generation for external OPA/OPS/OPL records.

Regression scenario: external expert feedback must create independent records
each time (no dedup/merge), with IDs derived from timestamps instead of
random UUID fragments.  This test verifies:

1. Repeated external OPA/OPS/OPL submissions produce distinct IDs containing
   timestamp suffixes (not UUID hex).
2. The full external expert flow works end-to-end: OPA -> OPS -> confirm ->
   OPL -> apply -> wiki entity updated.
"""

from __future__ import annotations

import os
import re
import time
from hashlib import sha256
from typing import TYPE_CHECKING, Iterator

import pytest

from wolfharness.capabilities.wiki.wiki_build_tools import WikiBuildTools


if TYPE_CHECKING:
    from pathlib import Path


# Timestamp suffix is %Y%m%d%H%M%S = 14 digits at the end of the ID.
_TIMESTAMP_RE = re.compile(r"\d{14}$")


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
def wiki_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allow_model_requests: Iterator[None],
) -> WikiBuildTools:
    """A WikiBuildTools instance over a fresh local wiki root.

    Set ``WIKI_STORAGE_BACKEND=viking`` (plus ``VIKING_API_KEY`` /
    ``VIKING_NAMESPACE``) to run against the real OpenViking service instead.
    """
    backend = os.environ.get("WIKI_STORAGE_BACKEND", "local")
    monkeypatch.setenv("WIKI_STORAGE_BACKEND", backend)
    wiki_root = tmp_path / "wiki"
    library_root = tmp_path / "library"
    wiki_root.mkdir()
    library_root.mkdir()
    return WikiBuildTools(wiki_root, library_root)


@pytest.mark.integration
class TestTimestampIDs:
    """External submissions mint distinct timestamp-suffixed IDs."""

    def test_opa_feedback_creates_distinct_timestamp_ids(
        self, wiki_tools: WikiBuildTools
    ) -> None:
        """Two feedback OPAs with same title/target get different timestamp IDs."""
        tools = wiki_tools
        store = tools.store
        target_uri = store.entity_uri("Device", None, "SY75C")
        store.write_entity(
            "Device", None, "SY75C",
            make_entity({"概述": "原始内容"}, title="SY75C"),
        )

        opa1 = tools.create_opa(
            title="测试反馈",
            description="专家反馈内容错误",
            category="feedback",
            target_uri=target_uri,
            finding="内容缺失或不完整",
            missing="关键信息不存在",
            recommendation="补充缺失内容",
            skip_dedupe_lookup=True,
        )
        # Timestamp has second precision; sleep to ensure a different value.
        time.sleep(1.1)
        opa2 = tools.create_opa(
            title="测试反馈",
            description="另一位专家反馈同样问题",
            category="feedback",
            target_uri=target_uri,
            finding="内容缺失或不完整",
            missing="关键信息不存在",
            recommendation="补充缺失内容",
            skip_dedupe_lookup=True,
        )

        id1, id2 = str(opa1["opa_id"]), str(opa2["opa_id"])
        assert id1 != id2
        assert _TIMESTAMP_RE.search(id1), f"OPA ID lacks timestamp: {id1}"
        assert _TIMESTAMP_RE.search(id2), f"OPA ID lacks timestamp: {id2}"

    def test_ops_external_ingest_creates_distinct_timestamp_ids(
        self, wiki_tools: WikiBuildTools
    ) -> None:
        """Two external OPS ingestions get different timestamp IDs."""
        tools = wiki_tools
        store = tools.store
        target_uri = store.entity_uri("Device", None, "SY75C")
        store.write_entity(
            "Device", None, "SY75C",
            make_entity({"概述": "原始内容"}, title="SY75C"),
        )

        opa = tools.create_opa(
            title="测试反馈",
            description="专家反馈",
            category="feedback",
            target_uri=target_uri,
            finding="内容缺失或不完整",
            missing="关键信息不存在",
            recommendation="补充缺失内容",
            skip_dedupe_lookup=True,
        )

        ops1 = tools.ingest_external_ops(
            parent_opa=str(opa["uri"]),
            title="专家修正",
            analysis="原内容有误",
            solution="修正为正确内容",
            evidence_uris=[target_uri],
            expert_id="expert1",
        )
        time.sleep(1.1)
        ops2 = tools.ingest_external_ops(
            parent_opa=str(opa["uri"]),
            title="专家修正",
            analysis="另一位专家也认为有误",
            solution="同样修正",
            evidence_uris=[target_uri],
            expert_id="expert2",
        )

        oid1, oid2 = str(ops1["ops_id"]), str(ops2["ops_id"])
        assert oid1 != oid2
        assert _TIMESTAMP_RE.search(oid1), f"OPS ID lacks timestamp: {oid1}"
        assert _TIMESTAMP_RE.search(oid2), f"OPS ID lacks timestamp: {oid2}"

    def test_opl_external_snapshot_creates_distinct_timestamp_ids(
        self, wiki_tools: WikiBuildTools
    ) -> None:
        """Two external OPL snapshots get different timestamp IDs."""
        tools = wiki_tools
        store = tools.store
        target_uri = store.entity_uri("Device", None, "SY75C")
        store.write_entity(
            "Device", None, "SY75C",
            make_entity({"概述": "原始内容"}, title="SY75C"),
        )

        opa = tools.create_opa(
            title="测试反馈",
            description="专家反馈",
            category="feedback",
            target_uri=target_uri,
            finding="内容缺失或不完整",
            missing="关键信息不存在",
            recommendation="补充缺失内容",
            skip_dedupe_lookup=True,
        )

        # Create and confirm two independent OPS records.
        ops1 = tools.ingest_external_ops(
            parent_opa=str(opa["uri"]),
            title="意见A",
            analysis="分析A",
            solution="方案A",
            evidence_uris=[target_uri],
            expert_id="expert1",
        )
        tools.update_ops(str(ops1["ops_id"]), status="confirmed", reviewed_by="reviewer")

        time.sleep(1.1)
        ops2 = tools.ingest_external_ops(
            parent_opa=str(opa["uri"]),
            title="意见B",
            analysis="分析B",
            solution="方案B",
            evidence_uris=[target_uri],
            expert_id="expert2",
        )
        tools.update_ops(str(ops2["ops_id"]), status="confirmed", reviewed_by="reviewer")

        current = store.read_entity("Device", None, "SY75C")
        assert current is not None
        expected_sha = sha256(current.encode("utf-8")).hexdigest()
        candidate = make_entity({"概述": "修正内容"}, title="SY75C")

        opl1 = tools.create_opl(
            parent_opa=str(opa["uri"]),
            ops_uris=[str(ops1["uri"])],
            title="提案A",
            proposal="提案内容A",
            rationale="理由A",
            evidence_uris=[target_uri],
            source_type="external_expert",
            candidate_content=candidate,
            expected_sha256=expected_sha,
        )
        time.sleep(1.1)
        opl2 = tools.create_opl(
            parent_opa=str(opa["uri"]),
            ops_uris=[str(ops2["uri"])],
            title="提案B",
            proposal="提案内容B",
            rationale="理由B",
            evidence_uris=[target_uri],
            source_type="external_expert",
            candidate_content=candidate,
            expected_sha256=expected_sha,
        )

        assert opl1["opl_id"] != opl2["opl_id"]
        assert _TIMESTAMP_RE.search(opl1["opl_id"]), f"OPL ID lacks timestamp: {opl1['opl_id']}"
        assert _TIMESTAMP_RE.search(opl2["opl_id"]), f"OPL ID lacks timestamp: {opl2['opl_id']}"


@pytest.mark.integration
class TestExternalExpertFlow:
    """Full external expert flow: OPA -> OPS -> confirm -> OPL -> apply -> wiki updated."""

    def test_full_flow_lands_expert_correction(self, wiki_tools: WikiBuildTools) -> None:
        """External expert correction flows through OPA/OPS/OPL and updates the wiki."""
        tools = wiki_tools
        store = tools.store
        target_uri = store.entity_uri("Device", None, "SY75C")

        # 1. Write initial entity.
        store.write_entity(
            "Device", None, "SY75C",
            make_entity({"常见故障及故障机理": "原始故障表", "诊断流程": "标准流程"}, title="SY75C"),
        )

        # 2. Compute expected_sha256 for optimistic locking.
        current = store.read_entity("Device", None, "SY75C")
        assert current is not None
        expected_sha = sha256(current.encode("utf-8")).hexdigest()

        # 3. Create OPA (feedback category -> always new record with timestamp).
        opa = tools.create_opa(
            title="专家反馈故障表缺失",
            description="常见故障及故障机理章节内容不完整",
            category="feedback",
            reason_code="content_missing",
            target_uri=target_uri,
            target_section="常见故障及故障机理",
            finding="内容缺失或不完整",
            missing="关键信息不存在",
            recommendation="补充缺失内容",
            skip_dedupe_lookup=True,
        )
        assert _TIMESTAMP_RE.search(str(opa["opa_id"]))

        # 4. Create OPS via external expert ingestion.
        candidate = make_entity(
            {"常见故障及故障机理": "专家修正:增加叶片磨损失效模式", "诊断流程": "标准流程"},
            title="SY75C",
        )
        ops = tools.ingest_external_ops(
            parent_opa=str(opa["uri"]),
            title="专家修正意见",
            analysis="原表缺少叶片磨损导致效率下降的失效模式",
            solution="补充叶片磨损失效模式描述",
            evidence_uris=[target_uri],
            expert_id="expert@sany.com",
            expert_name="张工",
            candidate_content=candidate,
            expected_sha256=expected_sha,
        )
        assert _TIMESTAMP_RE.search(str(ops["ops_id"]))
        assert ops["source_type"] == "external_expert"
        assert ops["status"] == "unconfirmed"

        # 5. Business reviewer confirms OPS.
        confirmed = tools.update_ops(
            str(ops["ops_id"]),
            status="confirmed",
            reviewed_by="reviewer@sany.com",
            review_notes="确认修正有效",
        )
        assert confirmed["status"] == "confirmed"

        # 6. Create OPL (external snapshot -> timestamp ID).
        opl = tools.create_opl(
            parent_opa=str(opa["uri"]),
            ops_uris=[str(ops["uri"])],
            title="知识修正提案",
            proposal="将专家修正的故障机理表应用到Wiki",
            rationale="原表缺失关键失效模式,专家补充后更完整",
            evidence_uris=[target_uri],
            source_type="external_expert",
            candidate_content=candidate,
            expected_sha256=expected_sha,
        )
        assert _TIMESTAMP_RE.search(str(opl["opl_id"]))
        assert opl["source_type"] == "external_expert"
        assert opl["apply_status"] == "not_applied"

        # 7. Apply OPL -> wiki entity updated.
        result = tools.apply_opl(str(opl["opl_id"]))
        assert result["apply_status"] == "applied"

        # 8. Verify wiki entity was updated.
        stored = store.read_entity("Device", None, "SY75C") or ""
        assert "专家修正:增加叶片磨损失效模式" in stored

    def test_repeated_opa_does_not_merge_into_same_file(
        self, wiki_tools: WikiBuildTools
    ) -> None:
        """Repeated feedback OPAs create separate files, not append to one."""
        tools = wiki_tools
        store = tools.store
        target_uri = store.entity_uri("Device", None, "SY75C")
        store.write_entity(
            "Device", None, "SY75C",
            make_entity({"概述": "原始内容"}, title="SY75C"),
        )

        # Submit the same feedback three times.
        uris: list[str] = []
        for i in range(3):
            opa = tools.create_opa(
                title="重复反馈测试",
                description=f"第{i + 1}次反馈同一问题",
                category="feedback",
                target_uri=target_uri,
                finding="内容缺失或不完整",
            missing="关键信息不存在",
            recommendation="补充缺失内容",
            skip_dedupe_lookup=True,
            )
            uris.append(str(opa["uri"]))
            if i < 2:
                time.sleep(1.1)

        # All three URIs must be distinct (three separate files).
        assert len(set(uris)) == 3, f"Expected 3 distinct OPA URIs, got: {uris}"

        # Each OPA file must be independently readable.
        for uri in uris:
            assert tools.read_resource(uri) is not None, f"OPA file not readable: {uri}"
