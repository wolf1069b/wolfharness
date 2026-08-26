"""models — 知识库核心数据模型.

对应 design.md §三 概念层与实体层设计。
包含统一的数据模型、抽取结果容器和自校验逻辑。

所有入库资源均使用 viking:// URI 协议（viking://resources/<namespace>/...），
通过 resource.json 维护 uri → 文件路径映射。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import re
from typing import Any, ClassVar
from urllib.parse import quote

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
    "source_incomplete",  # 源真没有（无品牌型号）→ 接外部 BOM/规格表
    "extraction_missed",  # 源有内容但管线没物化实体 → 修管线/补跑 extraction
    "relation_missed",  # 目标实体存在但没链接 → 补 link / rebuild_backlinks
    "param_unhosted",  # 有标准值但无 Component 载体 → 等 Component 物化后回填
    "manual_error",  # 手册内容错/跨章节矛盾 → 人工裁决
    "process_conflict",  # hook/任务规范冲突 → 规范对齐（当前 fact_conflict 多属此类）
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
            self.finding or "未单独填写；详见问题描述与证据。",
            "",
            "## 缺失点",
            "",
            self.missing or "未单独填写；由对应 audit issue 继续追踪。",
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
        lines += [
            f"> 状态：{zh_status(self.status)}（{self.status}）· 类型：{zh_category(self.category)}（{self.category}）· 原因：{zh_reason(self.reason_code)}（{self.reason_code}）· 关闭状态：{zh_closure(self.closure_status)}（{self.closure_status}）",
            "",
        ]
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
        solution = self.solution or "<!-- 待补录：请在此描述解决方案 -->"
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
                self.analysis or "待补充：说明为什么该建议能解决 OPA 中的冲突或缺失。",
                "",
                "## 检索凭证",
                "",
                f"- query: {self.retrieval_query}",
                *[f"- scope: {uri}" for uri in self.retrieval_scopes],
                *[f"- hit: {uri}" for uri in self.retrieval_hit_uris],
                "" if self.retrieval_hit_uris else "- hit: 无（已执行检索，保留 open_gap）",
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
                "> 状态：{}（{}）· 类型：{}（{}）· 应用状态：{}（{}）".format(
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
            "> 状态：已应用（applied）。OPL 已通过版本校验并应用到正式实体。"
            if self.status == "applied"
            else "> 状态：待确认（unconfirmed）。OPL 仅供后续人工/专家确认，不能直接作为正式实体内容发布。"
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
                self.proposal or "待补充：描述拟写入的知识变化。",
                "",
                "## 形成依据",
                "",
                self.rationale or "待补充：说明证据如何支持该提案。",
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


def _wiki_resource_hash(*parts: str) -> str:
    """生成确定性的 SHA256 哈希 ID（24 字符），用于 viking:// URI。."""
    raw = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def concept_kb_uri(concept_name: str) -> str:
    """生成概念的 viking:// URI。."""
    resource_id = _wiki_resource_hash("concept", concept_name)
    return f"{wiki_resources_root()}/{resource_id}"


def entity_kb_uri(entity_type: str, entity_name: str, model_id: str = "") -> str:
    """生成实体的 viking:// URI。."""
    resource_id = _wiki_resource_hash("entity", entity_type, entity_name, model_id)
    return f"{wiki_resources_root()}/{resource_id}"


# ── 来源引用 ─────────────────────────────────────────────────────────────────


@dataclass
class SourceRef:
    """来源引用，指向 library/ 中的原始章节。."""

    library_path: str  # e.g. "library/sy215/ch05/chapter.md"
    section: str  # e.g. "§5.2 液压泵结构与工作原理"
    page_range: str = ""  # e.g. "p112-114"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.library_path:
            errors.append("SourceRef.library_path 不能为空")
        if not self.section:
            errors.append("SourceRef.section 不能为空")
        return errors


# ── 概念层模型 ───────────────────────────────────────────────────────────────


@dataclass
class ConceptModel:
    """概念层数据模型 — 对应 concepts/{category}/{组件名}.md 文件。."""

    name: str  # 组件通用名, e.g. "液压泵"
    file_path: str  # wiki 中的相对路径, e.g. "concepts/液压泵/液压泵.md"
    kb_uri: str = ""  # source URI (viking://, hash-based, no CJK)
    overview: str = ""  # 概述
    aliases: list[str] = field(default_factory=list)  # 别称
    models: list[dict[str, Any]] = field(default_factory=list)  # 型号明细
    properties: dict[str, Any] = field(default_factory=dict)  # 结构化属性（schema 定义 + 自由扩展）
    relations: list[dict[str, Any]] = field(default_factory=list)  # schema-whitelisted typed edges
    references: list[Any] = field(
        default_factory=list
    )  # 引用列表 (str | dict with title/file_path/kb_uri)
    sources: list[SourceRef] = field(default_factory=list)  # 来源追踪

    def __post_init__(self) -> None:
        if not self.kb_uri and self.name:
            self.kb_uri = concept_kb_uri(self.name)

    def validate(self) -> list[str]:
        """校验概念模型完整性。."""
        errors: list[str] = []
        if not self.name:
            errors.append("ConceptModel.name 不能为空")
        if not self.file_path:
            errors.append(f"ConceptModel({self.name}).file_path 不能为空")
        if not self.kb_uri:
            errors.append(f"ConceptModel({self.name}).kb_uri 为空")
        if not self.overview:
            errors.append(f"ConceptModel({self.name}).overview 为空，建议补充详细描述")
        for src in self.sources:
            errors.extend(src.validate())
        return errors

    def to_markdown(
        self,
        *,
        model_id: str | None = None,
        referenced_by: list[str] | None = None,
    ) -> str:
        """生成概念页 Markdown，包含反向实体引用与来源证据。."""
        lines: list[str] = [f"# {self.name}", ""]

        if self.aliases:
            lines += [f"**别称**: {', '.join(self.aliases)}", ""]

        if self.overview:
            lines += ["## 用途说明", "", self.overview, ""]

        if self.properties:
            lines += ["## 结构化属性", ""]
            for key, value in self.properties.items():
                if isinstance(value, list):
                    lines.append(f"- **{key}**: {', '.join(str(v) for v in value)}")
                else:
                    lines.append(f"- **{key}**: {value}")
            lines.append("")

        if self.models:
            lines += ["## 各机型细化参数记录", ""]
            for model in self.models:
                model_name = model.get("型号", "未知型号")
                machine = model.get("配套机型", "")
                heading = f"### {machine} ({model_name})" if machine else f"### {model_name}"
                lines += [heading, ""]
                for key, value in model.items():
                    if key in {"型号", "配套机型"}:
                        continue
                    lines.append(f"- {key}: {value}")
                lines.append("")

        if self.relations:
            lines += ["## Schema 关系", ""]
            for relation in self.relations:
                relation_name = str(relation.get("relation") or "")
                target_type = str(relation.get("target_type") or "")
                target = str(relation.get("target") or "")
                lines.append(f"- {relation_name} → {target}（{target_type}）")
            lines.append("")

        if self.references:
            lines += ["## 引用本概念的页面", ""]
            for ref in self.references:
                # ref 格式: {"title": "xxx", "file_path": "yyy", "kb_uri": "zzz"}
                if isinstance(ref, dict):
                    title = ref.get("title", ref.get("file_path", ""))
                    uri = ref.get("kb_uri", "")
                    lines.append(f"- [{title}]({uri})" if uri else f"- {title}")
                else:
                    lines.append(f"- {ref}")
            lines.append("")

        if referenced_by:
            lines += ["## 被引用列表【概念对应实体】", ""]
            for entity_uri in referenced_by:
                lines.append(f"- {entity_uri}")
            lines.append("")

        if self.sources:
            lines += ["---", "", "## Footnotes 与来源追踪 【概念找原文】", ""]
            for i, src in enumerate(self.sources, 1):
                page = f" {src.page_range}" if src.page_range else ""
                lines.append(f"[^{i}]: {src.section}{page}。[查看原文]({src.library_path})")
            lines.append("")

        return "\n".join(lines)


# ── 实体层模型 ───────────────────────────────────────────────────────────────


@dataclass
class EntityModel:
    """实体层数据模型 — 对应 private/{model}/{type}/{name}.md 文件。."""

    entity_type: str  # "system", "dtc", "symptom", "testaction", etc.
    name: str  # e.g. "hydraulic-system"
    file_path: str  # wiki 中的相对路径
    model_id: str = ""
    kb_uri: str = ""  # source URI (viking://, hash-based, no CJK)
    title: str = ""
    aliases: list[str] = field(default_factory=list)  # 外文术语、厂家俗称、缩写
    summary: str = ""  # 一句话语义摘要（用于 L0/L1 导航）
    content: str = ""
    concepts_used: list[str] = field(default_factory=list)  # [[concepts/液压泵]]
    relations: list[dict[str, Any]] = field(default_factory=list)  # schema-whitelisted typed edges
    properties: dict[str, Any] = field(default_factory=dict)  # 结构化属性（schema 定义 + 自由扩展）
    sources: list[SourceRef] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.kb_uri and self.name:
            self.kb_uri = entity_kb_uri(self.entity_type, self.name, self.model_id)

    def validate(self) -> list[str]:
        """校验实体模型完整性。."""
        errors: list[str] = []
        if not self.name:
            errors.append("EntityModel.name 不能为空")
        if not self.entity_type:
            errors.append(f"EntityModel({self.name}).entity_type 不能为空")
        if not self.file_path:
            errors.append(f"EntityModel({self.name}).file_path 不能为空")
        if not self.kb_uri:
            errors.append(f"EntityModel({self.name}).kb_uri 为空")
        if not self.content:
            errors.append(f"EntityModel({self.name}).content 为空")
        for src in self.sources:
            errors.extend(src.validate())
        return errors

    def to_markdown(
        self,
        *,
        model_id: str | None = None,
        reverse_relations: dict[str, list[str]] | None = None,
        template: str = "",
    ) -> str:
        """生成实体页 Markdown。.

        When a template string is provided (from the schema entity type definition),
        the page is rendered following the template structure with header fields
        (机型/摘要/引用概念/出处) and content sections. Otherwise, falls back to
        the generic rendering with structured properties and concept references.
        """
        mid = model_id or self.model_id

        if template:
            return self._render_template(
                template, model_id=mid, reverse_relations=reverse_relations
            )

        lines: list[str] = []

        if self.title:
            lines += [f"# {self.title}", ""]

        if self.aliases:
            lines += [f"**别称**: {', '.join(self.aliases)}", ""]

        # ponytail: strip leading H1 from LLM content if it duplicates title
        body = self._strip_trailing_concept_refs(self.content)
        if self.title:
            stripped = body.lstrip()
            prefix = f"# {self.title}"
            if stripped.startswith(prefix):
                rest = stripped[len(prefix) :]
                if not rest or rest[0] in ("\n", "\r"):
                    body = rest.lstrip("\n")
        lines += [body, ""]

        if self.properties:
            lines += ["## 结构化属性", ""]
            for key, value in self.properties.items():
                if isinstance(value, list):
                    lines.append(f"- **{key}**: {', '.join(str(v) for v in value)}")
                else:
                    lines.append(f"- **{key}**: {value}")
            lines.append("")

        if self.concepts_used:
            lines += ["## 被引用概念", ""]
            for concept_ref in self.concepts_used:
                name = self._extract_concept_name(concept_ref)
                uri = concept_kb_uri(name)
                if mid:
                    direct_uri = f"{wiki_resources_root()}/{quote(mid, safe='')}/concepts/{quote(name, safe='')}.md"
                    lines.append(f"- [[{uri}]] ({direct_uri})")
                else:
                    lines.append(f"- [[{uri}]]")
            lines.append("")

        if self.relations:
            lines += ["## Schema 关系", ""]
            for relation in self.relations:
                relation_name = str(relation.get("relation") or "")
                target_type = str(relation.get("target_type") or "")
                target = str(relation.get("target") or "")
                inverse = str(relation.get("inverse") or "")
                label = f"{relation_name} → {target}"
                if target_type:
                    label += f"（{target_type}）"
                if inverse:
                    label += f"；反向关系：{inverse}"
                lines.append(f"- {label}")
            lines.append("")

        if reverse_relations:
            related = {
                path
                for paths in reverse_relations.values()
                for path in paths
                if path not in (self.file_path, self.kb_uri)
            }
            if related:
                lines += ["## 相关实体", ""]
                for rel_uri in sorted(related):
                    lines.append(f"- {rel_uri}")
                lines.append("")

        if self.sources:
            lines += ["---", "", "## Footnotes 与来源追踪", ""]
            for i, src in enumerate(self.sources, 1):
                page = f" {src.page_range}" if src.page_range else ""
                lines.append(f"[^{i}]: {src.section}{page}。[查看原文]({src.library_path})")
            lines.append("")

        return "\n".join(lines)

    def _render_template(
        self,
        template: str,
        *,
        model_id: str,
        reverse_relations: dict[str, list[str]] | None,
    ) -> str:
        """Render entity page using a schema-defined Markdown template.

        The template uses ``{placeholder}`` syntax. Known placeholders:
        ``{title}``, ``{model}``, ``{summary}``, ``{concepts_used}``,
        ``{source}``, ``{content_*}`` (mapped from properties), ``{related_entities}``,
        ``{related_system}``, ``{kb_uri}``, ``{code}``, ``{severity}``.
        Unknown ``{content_*}`` placeholders default to the entity's full content.
        """
        # Build concept reference string
        concepts_str = "、".join(self.concepts_used) if self.concepts_used else "—"

        # Build source string
        source_str = (
            "；".join(
                f"{src.section}{' ' + src.page_range if src.page_range else ''}"
                for src in self.sources
            )
            if self.sources
            else "—"
        )

        # Build related entities string from reverse relations
        related_str = ""
        if reverse_relations:
            related = sorted({
                path
                for paths in reverse_relations.values()
                for path in paths
                if path not in (self.file_path, self.kb_uri)
            })
            related_str = "\n".join(f"- {uri}" for uri in related) if related else ""
        if not related_str and self.relations:
            related_str = (
                "\n".join(
                    f"- {rel.get('relation', '')} → {rel.get('target', '')}"
                    for rel in self.relations
                    if rel.get("relation") != "member_of_concept"
                )
                or ""
            )

        # Build related system for DTC/Schematic
        related_system = ""
        for rel in self.relations:
            if rel.get("relation") == "belongs_to_system":
                related_system = f"[[private/{model_id}/system/{rel.get('target', '')}]]"
                break

        # Map content_* placeholders from properties
        render_vars: dict[str, str] = {
            "title": self.title or self.name,
            "model": model_id,
            "summary": self.summary or "",
            "concepts_used": concepts_str,
            "source": source_str,
            "related_entities": related_str or "—",
            "related_system": related_system or "—",
            "kb_uri": self.kb_uri,
            "code": str(self.properties.get("code", "")),
            "severity": str(self.properties.get("severity", "")),
        }

        # Map content_* placeholders: try properties first, then fall back to content
        # Extract all placeholder names from template
        import re

        placeholders = re.findall(r"\{([a-z_]+)\}", template)
        for ph in placeholders:
            if ph.startswith("content_"):
                prop_key = ph  # e.g. content_function → look for content_function property
                if prop_key in self.properties:
                    render_vars[ph] = str(self.properties[prop_key])
                elif ph == "content":
                    render_vars[ph] = self.content
                else:
                    # Try to find matching content section in the entity's content
                    render_vars[ph] = self._extract_content_section(ph.removeprefix("content_"))

        # If no content_* placeholders matched, use full content for generic "content"
        if "content" in placeholders and "content" not in render_vars:
            render_vars["content"] = self.content

        # Safely format template, leaving unknown placeholders empty
        result = self._safe_format(template, render_vars)
        return result.rstrip() + "\n"

    @staticmethod
    def _safe_format(template: str, variables: dict[str, str]) -> str:
        """Format template with variables, replacing unknown placeholders with empty string."""
        import re

        def replacer(match: re.Match[str]) -> str:
            key = match.group(1)
            return str(variables.get(key, ""))

        return re.sub(r"\{([a-z_]+)\}", replacer, template)

    # Map template placeholder suffixes to Chinese heading keywords
    _SECTION_KEYWORDS: ClassVar[dict[str, str]] = {
        "function": "功能|概述|原理",
        "overview": "功能|概述|原理",
        "composition": "组成|结构",
        "principle": "原理|工作",
        "conditions": "条件|工况",
        "description": "描述|说明",
        "trigger": "触发|条件",
        "reset": "清除|复位",
        "troubleshoot": "排查|步骤|诊断",
        "causes": "原因|可能",
        "repair": "维修|建议",
        "purpose": "目的|用途",
        "prerequisites": "前置|条件|准备",
        "tools": "工具",
        "steps": "步骤|操作",
        "criteria": "判定|标准",
        "notes": "注意|事项",
        "applicable": "适用|故障",
        "parts": "零件|部件",
        "connections": "连接|关系",
        "params": "参数|规格",
        "schedule": "周期|维护|计划",
        "safety": "安全|注意",
        "phenomenon": "现象|表现",
        "flow": "流程",
    }

    def _extract_content_section(self, section_name: str) -> str:
        """Try to extract a named section from the entity's content.

        Looks for a markdown heading matching section_name (via keyword mapping)
        and returns the content under it. Falls back to the full content.
        """
        import re

        keywords = self._SECTION_KEYWORDS.get(section_name, section_name)
        # Match: ## heading containing keywords, then capture until next ##+ heading or end
        heading_pattern = re.compile(
            rf"#+\s*[^\n]*(?:{keywords})[^\n]*\n((?:(?!#+\s)[^\n]*\n?)*)",
            re.IGNORECASE,
        )
        match = heading_pattern.search(self.content)
        if match:
            extracted = match.group(1).strip()
            if extracted:
                return extracted
        return self.content.strip()

    @staticmethod
    def _strip_trailing_concept_refs(content: str) -> str:
        lines = content.rstrip().splitlines()
        while lines:
            stripped = lines[-1].strip()
            if _CONCEPT_NAME_RE.fullmatch(stripped) or stripped in {"", "---"}:
                lines.pop()
            else:
                break
        return "\n".join(lines)

    @staticmethod
    def _extract_concept_name(ref: str) -> str:
        match = _CONCEPT_NAME_RE.search(ref)
        return match.group(1) if match else ref.strip()


# ── 抽取结果（统一容器） ─────────────────────────────────────────────────────


@dataclass
class ExtractionResult:
    """知识抽取结果 — 规则引擎与 LLM 抽取器共用。."""

    concepts: dict[str, ConceptModel] = field(default_factory=dict)
    entities: list[EntityModel] = field(default_factory=list)
    relations: dict[str, list[str]] = field(default_factory=dict)
    opas: list[OPAModel] = field(default_factory=list)

    def merge(
        self,
        other: ExtractionResult,
        *,
        relation_cardinalities: dict[tuple[str, str], str] | None = None,
    ) -> None:
        """将 other 的结果合并到 self（原地修改）。."""
        for name, concept in other.concepts.items():
            if name in self.concepts:
                existing_concept = self.concepts[name]
                existing_concept.sources.extend(concept.sources)
                existing_concept.aliases = list(
                    dict.fromkeys([*existing_concept.aliases, *concept.aliases])
                )
                if concept.overview and concept.overview != existing_concept.overview:
                    opa_id = f"opa-concept-{_wiki_resource_hash(name, 'overview')}"
                    if not any(item.opa_id == opa_id for item in self.opas):
                        self.opas.append(
                            OPAModel(
                                opa_id=opa_id,
                                title=f"概念 {name} 描述冲突",
                                description="同一概念在不同章节中出现了不同的用途或定义描述",
                                category="conflict",
                                scope="concept",
                                target_uri=existing_concept.kb_uri,
                                target_path=existing_concept.file_path,
                                evidence_uris=list(
                                    dict.fromkeys(
                                        src.library_path
                                        for src in [*existing_concept.sources, *concept.sources]
                                    ),
                                ),
                                status="pending",
                            ),
                        )
                for key, value in concept.properties.items():
                    if (
                        key in existing_concept.properties
                        and existing_concept.properties[key] != value
                    ):
                        opa_id = f"opa-concept-property-{_wiki_resource_hash(name, key)}"
                        if not any(item.opa_id == opa_id for item in self.opas):
                            self.opas.append(
                                OPAModel(
                                    opa_id=opa_id,
                                    title=f"概念 {name} 属性冲突",
                                    description=(
                                        f"属性 `{key}` 在不同章节中出现不同取值：{existing_concept.properties[key]!s}；{value!s}"
                                    ),
                                    category="conflict",
                                    scope="concept",
                                    target_uri=existing_concept.kb_uri,
                                    target_path=existing_concept.file_path,
                                    target_section=key,
                                    evidence_uris=list(
                                        dict.fromkeys(
                                            src.library_path
                                            for src in [*existing_concept.sources, *concept.sources]
                                        ),
                                    ),
                                    status="pending",
                                ),
                            )
                    existing_concept.properties[key] = value
                existing_concept.relations = [
                    *existing_concept.relations,
                    *[
                        relation
                        for relation in concept.relations
                        if relation not in existing_concept.relations
                    ],
                ]
                for model in concept.models:
                    self._record_model_conflicts(existing_concept, model)
                    if model not in existing_concept.models:
                        existing_concept.models.append(model)
            else:
                self.concepts[name] = concept

        self._detect_entity_conflicts(other.entities)
        existing_entities = {
            (entity.entity_type, entity.name, entity.model_id): entity for entity in self.entities
        }
        for incoming in other.entities:
            key = (incoming.entity_type, incoming.name, incoming.model_id)
            existing = existing_entities.get(key)
            if existing is None:
                self.entities.append(incoming)
                existing_entities[key] = incoming
                continue
            if incoming.content and incoming.content != existing.content:
                existing.content = (
                    f"{existing.content.rstrip()}\n\n---\n\n{incoming.content.lstrip()}"
                )
            if incoming.title and not existing.title:
                existing.title = incoming.title
            if incoming.summary and not existing.summary:
                existing.summary = incoming.summary
            existing.concepts_used = list(
                dict.fromkeys([*existing.concepts_used, *incoming.concepts_used])
            )
            existing.relations = [
                *existing.relations,
                *[
                    relation
                    for relation in incoming.relations
                    if relation not in existing.relations
                ],
            ]
            existing.properties = {**existing.properties, **incoming.properties}
            existing.sources.extend(incoming.sources)
        self.opas.extend(other.opas)

        for concept_name, entity_paths in other.relations.items():
            bucket = self.relations.setdefault(concept_name, [])
            for p in entity_paths:
                if p not in bucket:
                    bucket.append(p)

        if relation_cardinalities:
            self.enforce_relation_cardinality(relation_cardinalities)

    def enforce_relation_cardinality(
        self,
        relation_cardinalities: dict[tuple[str, str], str],
    ) -> None:
        """约束合并后的关系基数，并把违反 Schema 的边记录为 OPA。.

        单章节抽取时 ``one`` 关系可能合法，但跨章节合并会产生多个目标。
        这里保留稳定顺序中的第一个目标，同时把被裁剪的候选和证据写入 OPA，
        不静默丢失冲突信息。
        """
        for concept in self.concepts.values():
            self._enforce_owner_relations(
                owner_type="concept",
                owner_name=concept.name,
                owner_uri=concept.kb_uri,
                owner_path=concept.file_path,
                relations=concept.relations,
                sources=concept.sources,
                relation_cardinalities=relation_cardinalities,
            )
        for entity in self.entities:
            self._enforce_owner_relations(
                owner_type=entity.entity_type,
                owner_name=entity.name,
                owner_uri=entity.kb_uri,
                owner_path=entity.file_path,
                relations=entity.relations,
                sources=entity.sources,
                relation_cardinalities=relation_cardinalities,
                concepts_used=entity.concepts_used,
            )

    def _enforce_owner_relations(
        self,
        *,
        owner_type: str,
        owner_name: str,
        owner_uri: str,
        owner_path: str,
        relations: list[dict[str, Any]],
        sources: list[SourceRef],
        relation_cardinalities: dict[tuple[str, str], str],
        concepts_used: list[str] | None = None,
    ) -> None:
        unique: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        by_relation: dict[str, list[dict[str, Any]]] = {}
        for relation in relations:
            name = str(relation.get("relation") or "")
            target_type = str(relation.get("target_type") or "")
            target = str(relation.get("target") or "")
            key = (name, target_type, target)
            if key in seen:
                continue
            seen.add(key)
            by_relation.setdefault(name, []).append(relation)

        for relation_name, candidates in by_relation.items():
            cardinality = relation_cardinalities.get((owner_type, relation_name), "many")
            if cardinality != "one" or len(candidates) <= 1:
                unique.extend(candidates)
                continue
            kept = candidates[0]
            unique.append(kept)
            targets = [str(item.get("target") or "") for item in candidates]
            opa_id = f"opa-relation-{_wiki_resource_hash(owner_uri, relation_name)}"
            if not any(item.opa_id == opa_id for item in self.opas):
                self.opas.append(
                    OPAModel(
                        opa_id=opa_id,
                        title=f"{owner_name} 的 {relation_name} 关系基数冲突",
                        description=(
                            f"Schema 要求关系 `{relation_name}` cardinality=one，但跨章节抽取到多个目标：{'、'.join(targets)}；已保留首个目标，其余目标需要人工确认。"
                        ),
                        category="conflict",
                        scope="concept" if owner_type == "concept" else "entity",
                        target_uri=owner_uri,
                        target_path=owner_path,
                        target_section=relation_name,
                        evidence_uris=list(dict.fromkeys(src.library_path for src in sources)),
                        status="pending",
                    ),
                )
            if relation_name == "member_of_concept" and concepts_used is not None:
                kept_target = str(kept.get("target") or "")
                filtered_refs: list[str] = []
                for ref in concepts_used:
                    match = _CONCEPT_NAME_RE.search(ref)
                    if match is None or match.group(1) == kept_target:
                        filtered_refs.append(ref)
                concepts_used[:] = filtered_refs
        relations[:] = unique

    def _detect_entity_conflicts(self, incoming: list[EntityModel]) -> None:
        """Flag OPA when the same entity appears with different content across sources."""
        existing_map: dict[tuple[str, str, str], list[EntityModel]] = {}
        for ent in self.entities:
            existing_map.setdefault((ent.entity_type, ent.name, ent.model_id), []).append(ent)

        for new_ent in incoming:
            key = (new_ent.entity_type, new_ent.name, new_ent.model_id)
            siblings = existing_map.get(key)
            if not siblings:
                continue
            new_norm = _normalize_content(new_ent.content)
            for sib in siblings:
                if _normalize_content(sib.content) == new_norm:
                    continue
                opa_id = f"opa-ent-{_wiki_resource_hash(new_ent.entity_type, new_ent.name, new_ent.model_id)}"
                if any(o.opa_id == opa_id for o in self.opas):
                    continue
                evidence = list(
                    dict.fromkeys(
                        [
                            sib.kb_uri,
                            new_ent.kb_uri,
                            *(src.library_path for src in sib.sources),
                            *(src.library_path for src in new_ent.sources),
                        ],
                    ),
                )
                self.opas.append(
                    OPAModel(
                        opa_id=opa_id,
                        title=f"{new_ent.entity_type}/{new_ent.name} 内容冲突",
                        description=f"实体 {new_ent.name} 在不同章节中提取到不同内容",
                        category="conflict",
                        scope="entity",
                        target_uri=new_ent.kb_uri,
                        evidence_uris=evidence,
                        status="pending",
                    ),
                )
                break

    def detect_cross_concept_numerical_conflicts(self) -> None:
        """Flag OPA when different concepts report contradictory numerical values for the same parameter."""
        # ponytail: simple heuristic — group by key name, compare numeric values with ±10% tolerance
        param_map: dict[str, list[tuple[str, float, str]]] = {}
        for concept in self.concepts.values():
            for model in concept.models:
                for key, value in model.items():
                    if key in ("型号", "配套机型"):
                        continue
                    num = _parse_number(value)
                    if num is None:
                        continue
                    param_map.setdefault(key, []).append((concept.name, num, concept.kb_uri))

        for key, entries in param_map.items():
            if len(entries) < 2:
                continue
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    name1, val1, uri1 = entries[i]
                    name2, val2, uri2 = entries[j]
                    if name1 == name2:
                        continue
                    if val1 == 0 or val2 == 0:
                        if val1 != val2:
                            continue
                    elif abs(val1 - val2) / max(abs(val1), abs(val2)) <= 0.1:
                        continue
                    opa_id = f"opa-num-{_wiki_resource_hash(key)}"
                    if any(o.opa_id == opa_id for o in self.opas):
                        continue
                    self.opas.append(
                        OPAModel(
                            opa_id=opa_id,
                            title=f"参数 {key} 数值冲突",
                            description=f"概念 {name1} 记录 {key}={val1}, 概念 {name2} 记录 {key}={val2}",
                            category="numerical",
                            scope="concept",
                            evidence_uris=[uri1, uri2],
                            status="pending",
                        ),
                    )
                    break
                else:
                    continue
                break

    def _record_model_conflicts(self, concept: ConceptModel, incoming: dict[str, Any]) -> None:
        """Persist contradictory claims for the same machine/component variant."""
        identity_keys = ("配套机型", "型号")
        for existing in concept.models:
            if any(
                existing.get(key) and incoming.get(key) and existing.get(key) != incoming.get(key)
                for key in identity_keys
            ):
                continue
            for key in sorted((existing.keys() & incoming.keys()) - set(identity_keys)):
                old_value = existing.get(key)
                new_value = incoming.get(key)
                if old_value in (None, "") or new_value in (None, "") or old_value == new_value:
                    continue
                title = f"{concept.name}{key}取值冲突"
                if any(
                    opa.title == title and opa.description.endswith(f"{new_value!s}")
                    for opa in self.opas
                ):
                    continue
                self.opas.append(
                    OPAModel(
                        opa_id="",
                        title=title,
                        description=f"同一概念变体的 `{key}` 存在不同取值：{old_value!s}；{new_value!s}",
                        category="conflict",
                        scope="concept",
                        target_uri=concept.kb_uri,
                        target_path=concept.file_path,
                        target_section="各机型细化参数记录",
                        evidence_uris=list(
                            dict.fromkeys(src.library_path for src in concept.sources)
                        ),
                    ),
                )

    def validate(self) -> list[str]:
        """校验抽取结果中所有模型的完整性。."""
        errors: list[str] = []
        for concept in self.concepts.values():
            errors.extend(concept.validate())
        for entity in self.entities:
            errors.extend(entity.validate())
        return errors


# ── 构建结果 ─────────────────────────────────────────────────────────────────


@dataclass
class WikiBuildResult:
    """构建结果。."""

    model_id: str
    concepts: list[ConceptModel] = field(default_factory=list)
    entities: list[EntityModel] = field(default_factory=list)
    relations: dict[str, list[str]] = field(default_factory=dict)
    opas: list[OPAModel] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def concept_count(self) -> int:
        return len(self.concepts)

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def opa_count(self) -> int:
        return len(self.opas)

    def validate(self) -> list[str]:
        """校验构建结果。."""
        errors: list[str] = []
        if not self.model_id:
            errors.append("WikiBuildResult.model_id 不能为空")
        for concept in self.concepts:
            errors.extend(concept.validate())
        for entity in self.entities:
            errors.extend(entity.validate())
        return errors


# ── Resource Manifest 生成 ───────────────────────────────────────────────────


def opa_file_path(opa: OPAModel, model_id: str) -> str:
    """Compute the relative wiki path for an OPA file."""
    target_kind = "concepts" if opa.scope == "concept" else f"entities/{model_id}"
    return f"opa/{target_kind}/{opa.opa_id}.md"


def build_wiki_resource_manifest(
    *,
    model_id: str,
    concepts: list[ConceptModel],
    entities: list[EntityModel],
    opas: list[OPAModel] | None = None,
) -> dict[str, Any]:
    """构建 wiki 的 resource.json。.

    每个 concept/entity/opa 对应一个 resource 条目：
    - resource_id: viking:// URI 中的 hash ID
    - kind: "concept" / "entity_{entity_type}" / "opa"
    - uri: viking:// URI
    - path: 相对文件路径
    - mime_type: "text/markdown"
    - title: 页面标题

    下游使用方通过 resource.json 快速查 URI 表获得文件路径。
    """
    resources: list[dict[str, Any]] = []

    for concept in concepts:
        resource_id = concept.kb_uri.split("/")[-1] if concept.kb_uri else ""
        resources.append(
            {
                "resource_id": resource_id,
                "kind": "concept",
                "uri": concept.kb_uri,
                "path": concept.file_path,
                "mime_type": "text/markdown",
                "title": concept.name,
                "name": concept.name,
                "aliases": concept.aliases,
                "properties": concept.properties,
            },
        )

    for entity in entities:
        resource_id = entity.kb_uri.split("/")[-1] if entity.kb_uri else ""
        resources.append(
            {
                "resource_id": resource_id,
                "kind": f"entity_{entity.entity_type}",
                "uri": entity.kb_uri,
                "path": entity.file_path,
                "mime_type": "text/markdown",
                "title": entity.title or entity.name,
                "name": entity.name,
                "aliases": entity.aliases,
                "entity_type": entity.entity_type,
                "properties": entity.properties,
            },
        )

    if opas:
        resources.extend(
            {
                "resource_id": opa.opa_id,
                "kind": "opa",
                "uri": f"{wiki_resources_root()}/OP/{opa.opa_id}",
                "path": opa_file_path(opa, model_id),
                "mime_type": "text/markdown",
                "title": opa.title,
                "target_uri": opa.target_uri,
                "status": opa.status,
            }
            for opa in opas
        )

    return {
        "version": 1,
        "model_id": model_id,
        "uri": f"{wiki_resources_root()}/{model_id}/resource.json",
        "resources": resources,
    }


def build_wiki_resource_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """从 resource manifest 构建 resource_id → entry 索引。."""
    resources = manifest.get("resources")
    if not isinstance(resources, list):
        return {}
    index: dict[str, dict[str, Any]] = {}
    for entry in resources:
        if not isinstance(entry, dict):
            continue
        resource_id = entry.get("resource_id")
        if isinstance(resource_id, str) and resource_id:
            index[resource_id] = entry
    return index
