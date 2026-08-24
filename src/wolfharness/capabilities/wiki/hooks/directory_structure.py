"""Directory structure validation hook.

Validates that entity directory paths conform to design_729.md §2 KB
目录结构一览. Checks concept/class_name against expected patterns:

- Fault: 7 English class names (HydraulicFailure, MechanicalFailure, ...)
- Procedure: 10 English operation types (diagnosis, inspection, ...)
- DTC: 系列代号_控制器功能角色 (e.g. SY215_主控制器, SY365_发动机控制器)
- Component: logical type/variant path, NO 关重件/普通件 storage prefix
- Symptom: functional domain (not series name)
- Prohibited concepts: Series, System
"""

from __future__ import annotations

from wolfharness.capabilities.wiki.section_constants import (
    CONFIG_SPECIFIC_PATTERNS,
    MODEL_PREFIXES,
    MODEL_TOKEN_RE,
)

from .base import BaseHook, HookResult


# ── Design_717.md directory rules ─────────────────────────────────────────

_FAULT_CLASSES: frozenset[str] = frozenset({
    "HydraulicFailure",
    "MechanicalFailure",
    "ElectricalFailure",
    "ControlFailure",
    "FluidFailure",
    "ThermalFailure",
    "StructuralFailure",
})

_PROCEDURE_CLASSES: frozenset[str] = frozenset({
    "diagnosis",
    "inspection",
    "measurement",
    "maintenance",
    "removal",
    "installation",
    "replacement",
    "adjustment",
    "repair",
    "test",
})

_PROHIBITED_CONCEPTS: frozenset[str] = frozenset({
    "Series",
    "System",
})

_VALID_CONCEPTS: frozenset[str] = frozenset({
    "Device",
    "Component",
    "DTC",
    "Symptom",
    "Fault",
    "Procedure",
    "OP",
})

# Chinese class names that should be English (Fault)
_CHINESE_FAULT_INDICATORS: frozenset[str] = frozenset({
    "液压故障",
    "机械故障",
    "电气故障",
    "控制故障",
    "流体故障",
    "热故障",
    "结构故障",
    "传感器故障",
    "电气系统",
    "液压系统",
})

# Chinese class names that should be English (Procedure)
_CHINESE_PROCEDURE_INDICATORS: frozenset[str] = frozenset({
    "发动机",
    "操作规程",
    "拆装",
    "测试",
    "检查",
    "测量",
    "保养",
    "调整",
    "诊断",
    "维修",
    "测试与调整",
    "标定",
    "calibration",
    "disassembly",
})

# DTC class names that use the old MCU-SYC215 / ECU-QSB6.7 pattern.
# These should be 系列代号_控制器功能角色 (e.g. SY215_主控制器).
_DTC_LEGACY_PATTERNS: tuple[str, ...] = (
    "MCU-",
    "ECU-",
    "HCU-",
    "VCU-",
)

# Device model prefixes that must NOT appear in Component object_name.
# Component objects should be named by brand+model or mechanical type+specs,
# never by "机型+部件名" (e.g. "SY215C前泵压力传感器" is wrong).
# Centralized in wolfharness.capabilities.wiki.section_constants.MODEL_PREFIXES.


class DirectoryStructureHook(BaseHook):
    """Check that entity directory structure conforms to design_729.md.

    Validates:
    - Concept is a valid type (not Series/System)
    - Fault class_name is one of 7 English types
    - Procedure class_name is one of 10 English operation types
    - DTC class_name looks like a controller model
    - Component class_name must NOT contain a 关重件/ or 普通件/ prefix
    - Component object_name has no device model prefix (SY215C, SY365H, ...)
    """

    @property
    def name(self) -> str:
        return "directory_structure"

    def check(
        self,
        content: str,
        concept: str = "",
        class_name: str = "",
        object_name: str = "",
    ) -> HookResult:
        issues: list[str] = []

        # ── Prohibited concepts ───────────────────────────────────────────
        if concept in _PROHIBITED_CONCEPTS:
            return HookResult(
                hook_name=self.name,
                passed=False,
                message=(
                    f"Concept '{concept}' is prohibited — Series and System "
                    f"are not valid Concept types (design_729.md §2). "
                    f"Series aggregation is via Device/<系列>/<系列>.md, "
                    f"System overview is via raw chapters."
                ),
                severity="error",
            )

        if concept and concept not in _VALID_CONCEPTS:
            issues.append(
                f"Concept '{concept}' is not a recognized type. "
                f"Valid types: {', '.join(sorted(_VALID_CONCEPTS))}.",
            )

        # ── Fault: must use 7 English class names ─────────────────────────
        if concept == "Fault":
            if not class_name:
                issues.append("Fault entity has no class_name.")
            elif class_name in _CHINESE_FAULT_INDICATORS:
                issues.append(
                    f"Fault class_name '{class_name}' is Chinese — must use "
                    f"one of 7 English types: "
                    f"{', '.join(sorted(_FAULT_CLASSES))}.",
                )
            elif class_name not in _FAULT_CLASSES:
                issues.append(
                    f"Fault class_name '{class_name}' is not one of the 7 "
                    f"valid English failure types: "
                    f"{', '.join(sorted(_FAULT_CLASSES))}.",
                )

        # ── Procedure: must use 10 English operation types ────────────────
        if concept == "Procedure":
            if not class_name:
                issues.append("Procedure entity has no class_name.")
            elif class_name in _CHINESE_PROCEDURE_INDICATORS:
                issues.append(
                    f"Procedure class_name '{class_name}' is Chinese — must use "
                    f"one of 10 English operation types: "
                    f"{', '.join(sorted(_PROCEDURE_CLASSES))}.",
                )
            elif class_name not in _PROCEDURE_CLASSES:
                issues.append(
                    f"Procedure class_name '{class_name}' is not one of the 10 "
                    f"valid English operation types: "
                    f"{', '.join(sorted(_PROCEDURE_CLASSES))}.",
                )

        # ── DTC: class_name must be 系列代号_控制器功能角色 ───────────────
        if concept == "DTC":
            if not class_name:
                issues.append("DTC entity has no class_name.")
            elif MODEL_TOKEN_RE.match(class_name) is not None:
                issues.append(
                    f"DTC class_name '{class_name}' is a device series/model, "
                    f"not a controller. Use format: 系列代号_控制器功能角色 "
                    f"(e.g. SY215_主控制器, SY365_发动机控制器).",
                )
            elif any(class_name.startswith(p) for p in _DTC_LEGACY_PATTERNS):
                issues.append(
                    f"DTC class_name '{class_name}' uses legacy MCU/ECU/HCU "
                    f"naming — must use format: 系列代号_控制器功能角色 "
                    f"(e.g. SY215_主控制器, NOT MCU-SYC215). "
                    f"Controllers are identified by functional role from the "
                    f"manual, not by inventing model names with device codes.",
                )
            elif "_" not in class_name:
                issues.append(
                    f"DTC class_name '{class_name}' missing series prefix — "
                    f"must use format: 系列代号_控制器功能角色 "
                    f"(e.g. SY215_主控制器, SY365_主控制器).",
                )

        # ── Component: class_name must be pure type path, NO legacy prefix ─
        if concept == "Component":
            if not class_name:
                issues.append("Component entity has no class_name.")
            elif class_name.startswith(("关重件/", "普通件/")):
                issues.append(
                    f"Component class_name '{class_name}' contains a legacy "
                    f"storage prefix. Pass the logical assembly-path such as "
                    f"'主泵' or '主泵/双联柱塞式', not '关重件/主泵'. "
                    f"The 关重件/普通件 tier routing was removed; a Component "
                    f"folder is its logical class_name path.",
                )
            if object_name:
                bad_prefix = next(
                    (p for p in MODEL_PREFIXES if object_name.upper().startswith(p)),
                    None,
                )
                if bad_prefix:
                    issues.append(
                        f"Component object_name '{object_name}' starts with "
                        f"device model prefix '{bad_prefix}' — must use "
                        f"brand+model (e.g. '川崎K3V112') or mechanical type"
                        f"+specs (e.g. '往复柱塞式液压油缸Φ135×95×1490'), "
                        f"not '机型+部件名'.",
                    )

        # ── Symptom: class_name should be a functional domain ─────────────
        if concept == "Symptom":
            if not class_name:
                issues.append("Symptom entity has no class_name.")
            elif MODEL_TOKEN_RE.match(class_name) is not None:
                issues.append(
                    f"Symptom class_name '{class_name}' looks like a device "
                    f"model — should be a functional domain like "
                    f"动力与发动机, 液压系统, 电气系统.",
                )

            # ── Symptom index.md: 产品配置差异 must be a redirect, not inline ─
            # design_729.md §3.6: index.md's 产品配置差异 section should be
            # a one-liner redirect to profile/. Config-specific content
            # (model names, pressure values, component types) belongs in
            # profile/<profile_id>.md.
            if "## 产品配置差异" in content:
                # Extract the section content between ## 产品配置差异 and next ##
                lines = content.splitlines()
                in_section = False
                section_lines: list[str] = []
                for line in lines:
                    if line.strip() == "## 产品配置差异":
                        in_section = True
                        continue
                    if in_section and line.startswith("## "):
                        break
                    if in_section:
                        section_lines.append(line)

                section_text = "\n".join(section_lines).strip()
                # The canonical redirect is ~1-2 sentences. If it contains
                # specific model codes, component names, or numeric specs,
                # it's inline config content that belongs in a Profile.
                # Patterns are centralized in section_constants.CONFIG_SPECIFIC_PATTERNS.
                found_patterns = [p for p in CONFIG_SPECIFIC_PATTERNS if p in section_text]
                if found_patterns:
                    issues.append(
                        f"Symptom index.md '## 产品配置差异' contains "
                        f"config-specific content ({', '.join(found_patterns[:3])}). "
                        f"This belongs in profile/<profile_id>.md, not index.md. "
                        f"index.md should only have a redirect sentence like: "
                        f"'不同配置下的失效机理和诊断路径存在差异，详见对应 Symptom Profile。' "
                        f"(design_729.md §3.6).",
                    )

        if issues:
            return HookResult(
                hook_name=self.name,
                passed=False,
                message=" | ".join(issues),
                severity="error",
            )

        return HookResult(
            hook_name=self.name,
            passed=True,
            message=f"Directory structure OK for {concept}/{class_name}/{object_name}.",
        )
