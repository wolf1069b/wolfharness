"""models — 知识库核心数据模型.

对应 design.md §三 概念层与实体层设计。
包含统一的数据模型、抽取结果容器和自校验逻辑。

所有入库资源均使用 viking:// URI 协议(viking://resources/<namespace>/...),
通过 resource.json 维护 uri → 文件路径映射。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re

from wolfharness.capabilities.wiki.namespaces import wiki_resources_root


logger = logging.getLogger(__name__)

_CONCEPT_NAME_RE = re.compile(r"\[\[concepts/([^\]]+)\]\]")

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

OPA_REASON_CODES: tuple[str, ...] = (
    # Coarse reasons — set at OPA creation, before retrieval triage.
    "content_missing",
    "fact_conflict",
    "expert_feedback",
    # Fine reasons — OPS refines into these after retrieval; each maps to a
    # distinct repair path, documented inline because the names alone don't
    # reveal the fix direction.
    "source_incomplete",  # 源真没有(无品牌型号)→ 接外部 BOM/规格表
    "extraction_missed",  # 源有内容但管线没物化实体 → 修管线/补跑 extraction
    "relation_missed",  # 目标实体存在但没链接 → 补 link / rebuild_backlinks
    "param_unhosted",  # 有标准值但无 Component 载体 → 等 Component 物化后回填
    "manual_error",  # 手册内容错/跨章节矛盾 → 人工裁决
    "process_conflict",  # hook/任务规范冲突 → 规范对齐(当前 fact_conflict 多属此类)
    "hallucination",  # 模型捏造 → 丢弃/修正
)

# Fine-grained reasons an OPS may refine a coarse OPA into after retrieval.
OPA_REASON_FINE_CODES: frozenset[str] = frozenset(
    {
        "source_incomplete",
        "extraction_missed",
        "relation_missed",
        "param_unhosted",
        "manual_error",
        "process_conflict",
        "hallucination",
    },
)

OPA_CLOSURE_STATUSES: tuple[str, ...] = ("open", "deferred", "closed")


def infer_opa_reason_code(category: str) -> str:
    """Map the legacy OPA category to a stable machine-readable reason."""
    if category.strip().lower() in {"feedback", "expert_feedback"}:
        return "expert_feedback"
    return "fact_conflict" if category.strip().lower() == "conflict" else "content_missing"


def _normalize_content(text: str) -> str:
    """Normalize content for comparison: strip whitespace, lowercase first 200 chars."""
    return text.strip()[:200].casefold()


def _parse_number(value: object) -> float | None:
    """Extract the first numeric value from a string or numeric input."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    match = _NUM_RE.search(value)
    return float(match.group()) if match else None


def _yaml_quote(value: str) -> str:
    """Encode a user/LLM-controlled scalar safely for YAML frontmatter."""
    return json.dumps(value, ensure_ascii=False)


def _fm(value: object) -> str | None:
    """Render a YAML frontmatter field, or ``None`` to omit empty fields.

    Empty strings and empty lists are omitted so expert-facing files are
    not padded with ``field: ""`` noise.
    """
    if isinstance(value, str):
        return None if not value.strip() else _yaml_quote(value)
    if isinstance(value, (list, dict)) and not value:
        return None
    if isinstance(value, bool) or value is None:
        return str(value)
    return str(value)


def _fm_lines(fields: list[tuple[str, object]]) -> list[str]:
    """Render ``(key, value)`` pairs as ``key: value`` frontmatter lines."""
    return [f"{key}: {rendered}" for key, value in fields if (rendered := _fm(value)) is not None]


_ZH_STATUS: dict[str, str] = {
    "pending": "待处理",
    "resolved": "已处理",
    "rejected": "已拒绝",
    "superseded": "已取代",
    "unconfirmed": "待确认",
    "confirmed": "已确认",
    "applied": "已应用",
    "archived": "已归档",
}
_ZH_CLOSURE: dict[str, str] = {
    "open": "未关闭",
    "deferred": "延迟处理",
    "closed": "已关闭",
}
_ZH_CATEGORY: dict[str, str] = {
    "gap": "内容缺失",
    "conflict": "知识冲突",
    "feedback": "专家反馈",
}
_ZH_REASON: dict[str, str] = {
    "content_missing": "内容缺失",
    "fact_conflict": "事实冲突",
    "expert_feedback": "专家反馈",
    "source_incomplete": "来源不完整",
    "extraction_missed": "抽取遗漏",
    "relation_missed": "关系缺失",
    "param_unhosted": "参数无载体",
    "manual_error": "手册错误",
    "process_conflict": "流程冲突",
    "hallucination": "模型捏造",
}
_ZH_APPLY: dict[str, str] = {
    "not_ready": "未就绪",
    "not_applied": "未应用",
    "not_applicable": "不适用",
    "applied": "已应用",
    "needs_review": "待人工确认",
    "failed": "失败",
}


def zh_status(value: str) -> str:
    return _ZH_STATUS.get(value.strip().lower(), value)


def zh_closure(value: str) -> str:
    return _ZH_CLOSURE.get(value.strip().lower(), value)


def zh_category(value: str) -> str:
    return _ZH_CATEGORY.get(value.strip().lower(), value)


def zh_reason(value: str) -> str:
    return _ZH_REASON.get(value.strip().lower(), value)


def zh_apply(value: str) -> str:
    return _ZH_APPLY.get(value.strip().lower(), value)


# ── OPA/OPS feedback models ──────────────────────────────────────────────────


@dataclass
class OPAModel:
    """A single unresolved knowledge conflict or inconsistency."""

    opa_id: str
    title: str
    description: str
    human_key: str = ""
    category: str = "conflict"
    reason_code: str = ""
    scope: str = "entity"
    subtype: str = "wiki_error"  # wiki_error | category_error | case_feedback | user_feedback
    target_uri: str = ""
    target_path: str = ""
    target_section: str = ""
    source_chapter: str = ""
    evidence_uris: list[str] = field(default_factory=list)
    status: str = "pending"
    solution: str = ""
    finding: str = ""
    missing: str = ""
    recommendation: str = ""
    related_uris: list[str] = field(default_factory=list)
    report_count: int = 1
    dedupe_key: str = ""
    build_id: str = ""
    # closure_status is the substantive gap state, independent of the
    # administrative `status` (pending/resolved/...).  `resolved` no longer
    # implies the gap is closed — a record can be resolved (admin finalized)
    # yet still closure_status=open (gap unfixed). The finalize gate keys off
    # closure_status, not status, so status=resolved cannot bypass it.
    closure_status: str = "open"  # open | deferred | closed
    closure_reason: str = ""

    def __post_init__(self) -> None:
        """Keep records created by older callers readable and classifiable."""
        if not self.reason_code:
            self.reason_code = infer_opa_reason_code(self.category)
        if self.reason_code not in OPA_REASON_CODES:
            raise ValueError(f"Unsupported OPA reason_code: {self.reason_code!r}")
        if self.closure_status not in OPA_CLOSURE_STATUSES:
            raise ValueError(f"Unsupported OPA closure_status: {self.closure_status!r}")

    def to_markdown(self, *, ops_uri: str = "") -> str:
        """Render a durable OPA record with KB-addressable evidence."""
        evidence = (
            "\n".join(f"- [{uri}]({uri})" for uri in self.evidence_uris) or "- 尚无可解析证据 URI"
        )
        frontmatter = ["---", f"id: {_yaml_quote(self.opa_id)}"]
        frontmatter += _fm_lines(
            [
                ("category", self.category),
                ("reason_code", self.reason_code),
                ("status", self.status),
                ("target_uri", self.target_uri),
                ("target_section", self.target_section),
                ("scope", self.scope),
                ("subtype", self.subtype),
                ("source_chapter", self.source_chapter),
                ("evidence_uris", self.evidence_uris),
                ("related_uris", self.related_uris),
                ("report_count", self.report_count),
                ("dedupe_key", self.dedupe_key),
                ("build_id", self.build_id),
                ("closure_status", self.closure_status),
                ("closure_reason", self.closure_reason),
            ],
        )
        frontmatter.append("---")

        lines = [
            *frontmatter,
            "",
            f"# {self.title}",
            "",
            "## 问题描述",
            "",
            self.description,
            "",
            "## 冲突点",
            "",
            self.finding or "未单独填写;详见问题描述与证据。",
            "",
            "## 缺失点",
            "",
            self.missing or "未单独填写;由对应 audit issue 继续追踪。",
            "",
            "## 建议",
            "",
            self.recommendation or self.solution or "待 conductor 裁决。",
            "",
            "## 证据",
            "",
            evidence,
            "",
        ]
        if self.related_uris:
            lines += [
                "## 关联引用",
                "",
                *[f"- {uri}" for uri in self.related_uris],
                "",
            ]
        if self.solution:
            ops_id = self.opa_id.replace("opa-", "ops-", 1)
            # Fallback when caller has no store root_uri in scope; real callers pass ops_uri.
            linked_ops_uri = ops_uri or f"{wiki_resources_root()}/OP/{ops_id}"
            lines += [
                "## 关联解决方案",
                "",
                f"- [对应 OPS]({linked_ops_uri})",
                "",
            ]
        status_line = (
            f"> 状态:{zh_status(self.status)}({self.status})"
            f"· 类型:{zh_category(self.category)}({self.category})"
            f"· 原因:{zh_reason(self.reason_code)}({self.reason_code})"
            f"· 关闭状态:{zh_closure(self.closure_status)}({self.closure_status})"
        )
        lines += [status_line, ""]
        return "\n".join(lines)


@dataclass
class OPSModel:
    """An expert remediation suggestion attached to one OPA record."""

    ops_id: str
    parent_opa: str
    title: str = ""
    status: str = "unconfirmed"
    target_uri: str = ""
    solution: str = ""
    analysis: str = ""
    retrieval_query: str = ""
    retrieval_scopes: list[str] = field(default_factory=list)
    retrieval_hit_uris: list[str] = field(default_factory=list)
    retrieval_used_uris: list[str] = field(default_factory=list)
    evidence_uris: list[str] = field(default_factory=list)
    related_uris: list[str] = field(default_factory=list)
    candidate_content: str = ""
    candidate_operations: list[dict[str, object]] = field(default_factory=list)
    expected_sha256: str = ""
    source_type: str = "pipeline"
    reviewed_by: str = ""
    review_notes: str = ""
    apply_status: str = "not_ready"
    apply_error: str = ""
    applied_at: str = ""
    applied_entity_sha256: str = ""

    def to_markdown(self) -> str:
        """Render a source-backed OPS page."""
        solution = self.solution or "<!-- 待补录:请在此描述解决方案 -->"
        evidence = "\n".join(f"- [{uri}]({uri})" for uri in self.evidence_uris) or "- 尚无证据"
        frontmatter = ["---", f"id: {_yaml_quote(self.ops_id)}"]
        frontmatter += _fm_lines(
            [
                ("parent_opa", self.parent_opa),
                ("status", self.status),
                ("target_uri", self.target_uri),
                ("retrieval_query", self.retrieval_query),
                ("retrieval_scopes", self.retrieval_scopes),
                ("retrieval_hit_uris", self.retrieval_hit_uris),
                ("retrieval_used_uris", self.retrieval_used_uris),
                ("evidence_uris", self.evidence_uris),
                ("related_uris", self.related_uris),
                ("candidate_content", self.candidate_content),
                ("candidate_operations", self.candidate_operations),
                ("expected_sha256", self.expected_sha256),
                ("source_type", self.source_type),
                ("reviewed_by", self.reviewed_by),
                ("review_notes", self.review_notes),
                ("apply_status", self.apply_status),
                ("apply_error", self.apply_error),
                ("applied_at", self.applied_at),
                ("applied_entity_sha256", self.applied_entity_sha256),
            ],
        )
        frontmatter.append("---")
        return "\n".join(
            [
                *frontmatter,
                "",
                f"# {self.title or self.ops_id}",
                "",
                "## 关联问题",
                "",
                f"- [OPA]({self.parent_opa})",
                "",
                "## 专家分析",
                "",
                self.analysis or "待补充:说明为什么该建议能解决 OPA 中的冲突或缺失。",
                "",
                "## 检索凭证",
                "",
                f"- query: {self.retrieval_query}",
                *[f"- scope: {uri}" for uri in self.retrieval_scopes],
                *[f"- hit: {uri}" for uri in self.retrieval_hit_uris],
                "" if self.retrieval_hit_uris else "- hit: 无(已执行检索,保留 open_gap)",
                *[f"- used: {uri}" for uri in self.retrieval_used_uris],
                "",
                "## 解决方案描述",
                "",
                solution,
                "",
                "## 证据引用",
                "",
                evidence,
                "",
                "## 关联资源",
                "",
                *[f"- {uri}" for uri in self.related_uris],
                "" if self.related_uris else "- 无",
                "",
                "> 状态:{}({})· 类型:{}({})· 应用状态:{}({})".format(
                    zh_status(self.status),
                    self.status,
                    "管线生成" if self.source_type == "pipeline" else "外部专家",
                    self.source_type,
                    zh_apply(self.apply_status),
                    self.apply_status,
                ),
                "",
            ],
        )


@dataclass
class OPLModel:
    """A proposed knowledge update formed from the OP chain or an expert."""

    opl_id: str
    title: str
    parent_opa: str
    ops_uris: list[str] = field(default_factory=list)
    target_uri: str = ""
    status: str = "unconfirmed"
    proposal: str = ""
    rationale: str = ""
    evidence_uris: list[str] = field(default_factory=list)
    related_uris: list[str] = field(default_factory=list)
    source_type: str = "pipeline"
    expert_id: str = ""
    expert_name: str = ""
    source_uri: str = ""
    target_concept: str = ""
    target_class_name: str = ""
    target_object_name: str = ""
    expected_sha256: str = ""
    candidate_content: str = ""
    candidate_operations: list[dict[str, object]] = field(default_factory=list)
    apply_status: str = "not_applied"
    apply_error: str = ""
    applied_at: str = ""
    applied_entity_sha256: str = ""
    archive_reason: str = ""

    def to_markdown(self) -> str:
        """Render an OPL proposal with optional machine-applicable update data."""
        evidence = "\n".join(f"- [{uri}]({uri})" for uri in self.evidence_uris) or "- 尚无证据"
        ops = "\n".join(f"- [{uri}]({uri})" for uri in self.ops_uris) or "- 尚无 OPS"
        related = "\n".join(f"- {uri}" for uri in self.related_uris) or "- 无"
        footer = (
            "> 状态:已应用(applied)。OPL 已通过版本校验并应用到正式实体。"
            if self.status == "applied"
            else (
                "> 状态:待确认(unconfirmed)。OPL 仅供后续人工/专家确认,"
                "不能直接作为正式实体内容发布。"
            )
        )
        frontmatter = ["---", f"id: {_yaml_quote(self.opl_id)}"]
        frontmatter += _fm_lines(
            [
                ("parent_opa", self.parent_opa),
                ("ops_uris", self.ops_uris),
                ("target_uri", self.target_uri),
                ("status", self.status),
                ("evidence_uris", self.evidence_uris),
                ("related_uris", self.related_uris),
                ("source_type", self.source_type),
                ("expert_id", self.expert_id),
                ("expert_name", self.expert_name),
                ("source_uri", self.source_uri),
                ("target_concept", self.target_concept),
                ("target_class_name", self.target_class_name),
                ("target_object_name", self.target_object_name),
                ("expected_sha256", self.expected_sha256),
                ("candidate_content", self.candidate_content),
                ("candidate_operations", self.candidate_operations),
                ("apply_status", self.apply_status),
                ("apply_error", self.apply_error),
                ("applied_at", self.applied_at),
                ("applied_entity_sha256", self.applied_entity_sha256),
                ("archive_reason", self.archive_reason),
            ],
        )
        frontmatter.append("---")
        return "\n".join(
            [
                *frontmatter,
                "",
                f"# {self.title or self.opl_id}",
                "",
                "## 关联 OPA",
                "",
                f"- [OPA]({self.parent_opa})",
                "",
                "## 关联 OPS",
                "",
                ops,
                "",
                "## 初版知识提案",
                "",
                self.proposal or "待补充:描述拟写入的知识变化。",
                "",
                "## 形成依据",
                "",
                self.rationale or "待补充:说明证据如何支持该提案。",
                "",
                "## 证据引用",
                "",
                evidence,
                "",
                "## 关联资源",
                "",
                related,
                "",
                footer,
                "",
            ],
        )


# ── URI 工具 ────────────────────────────────────────────────────────────────
