"""Centralized section name constants for validation hooks.

Single source of truth for section headings used across the wiki build pipeline.
Currently populated from default_schema.yaml section definitions.
To adapt for a different manual/schema, update these constants.
"""

import re


# --- Model detection patterns ---
# Generic pattern: uppercase letters followed by digits + optional alphanumeric suffix
# Covers SANY (SY/SE/SK), Komatsu (PC), Hitachi (ZX/EX), Caterpillar (CAT),
# Kobelco (SK), Doosan (DX), Volvo (EC), Hyundai (HD), JCB, Liebherr (R), etc.
MODEL_TOKEN_RE = re.compile(r"^[A-Z]{2,4}\d+[A-Z0-9-]*$", re.IGNORECASE)

# Known model prefixes for stricter validation (extendable)
MODEL_PREFIXES: tuple[str, ...] = (
    "SY",
    "SE",
    "SK",
    "PC",
    "EX",
    "ZX",
    "CAT",
    "HD",
    "DX",
    "EC",
    "JCB",
    "R",
)

# Engine/controller codes that indicate config-specific content
CONFIG_SPECIFIC_PATTERNS: tuple[str, ...] = (
    "QSB",
    "4M50",
    "SC9D",
    "CM2150",
    "MCU",
    "ECU",
    "K3V",
    "K5V",
    "KMX",
    "SAA",
    "S6K",
    "4TNV",
    "J05",
    "D1146",
)

# --- Section heading constants (from default_schema.yaml) ---
SECTION_MECHANISM = "工作机理"
SECTION_OVERVIEW = "总成概览"
SECTION_SOURCE = "来源"
SECTION_FAILURE_MECHANISM = "失效机理"
SECTION_COMMON_FAULTS = "常见故障及故障机理"
SECTION_IMPACT_SCOPE = "影响范围"
SECTION_VERIFICATION = "验证方法"
SECTION_REPAIR_METHOD = "修复方式"
SECTION_ASSOCIATED_SYMPTOMS = "关联故障现象"
SECTION_POSSIBLE_FAILURE = "可能失效机理"
SECTION_DIAGNOSTIC_FLOW = "推荐诊断流程"
SECTION_DIAGNOSIS_FLOW = "诊断流程"
SECTION_CONTROLLER_IDENTITY = "控制器身份"
SECTION_SYSTEM_CHAPTERS = "系统章节引用"
SECTION_OPERATION_STEPS = "操作步骤"
SECTION_PREREQUISITES = "执行前提"
SECTION_REQUIRED_TOOLS = "所需工具"
SECTION_JUDGMENT_CRITERIA = "判定标准"
SECTION_COMMON_FAILURE_MODES = "常见失效模式"
SECTION_DISASSEMBLY_STEPS = "拆装步骤"

# Admin sections (low-value chapters to skip)
ADMIN_SECTION_KEYWORDS: tuple[str, ...] = (
    "安全",
    "前言",
    "cover",
    "intro",
    "safety",
    "foreword",
    "preface",
)

# Placeholder/gap text patterns (regex-escaped)
GAP_RE = r"open_gap|来源未说明|待补充|来源缺失|未物化|未提供|未确认|未说明|未知"
PLACEHOLDER_TEXT_RE = r"见来源|未提供|待补充|暂无|无资料|N/A|TODO|TBD"
