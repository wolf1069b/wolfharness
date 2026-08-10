"""Configuration model for DynamicContextPruningCapability.

Uses Pydantic v2 ``BaseModel`` for YAML/JSON deserialization, field
validation, and cross-field validation via ``@model_validator``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wolfharness.capabilities.dcp.state import WatermarkLevel


class DCPConfig(BaseModel):
    """Configuration for DynamicContextPruningCapability.

    Controls thresholds, tool protection, nudging behavior, and
    deduplication settings used by context-pruning strategies during a
    session.

    Attributes:
        enabled: Whether the capability is active.  When ``False``, all
            hooks are no-ops.
        expose_tools: Whether to expose prune/distill/decompress tools.
        info_threshold: Context pressure ratio for INFO watermark.
        warning_threshold: Context pressure ratio for WARNING watermark.
        critical_threshold: Context pressure ratio for CRITICAL watermark.
        max_context_tokens: Maximum context window size in tokens.
        inject_role: Role for ``<prunable-tools>`` list injection.
        nudge_role: Role for nudge text injection.
        nudge_turn_frequency: Number of turns between nudge injections.
        nudge_step_frequency: Number of tool-call steps between nudge
            injections (0 to disable step-based nudges).
        auto_dedup: Whether to auto-deduplicate at the strategy threshold.
        auto_strategy_threshold: Watermark level at which auto-strategies
            begin running.
        purge_error_steps: Tool-call steps before error tool calls are purged.
        step_protection: Recent tool-call steps protected from pruning.
        protected_tool_patterns: Glob patterns for protected tools.
        protected_tools: Explicit protected tool names (expanded from
            patterns + user-specified entries).
        meta_tool_retention: How many recent meta-tool returns to keep.
        clear_thinking_enabled: Whether the ``clear_thinking`` parameter
            on the prune tool is active.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool = True
    expose_tools: bool = True
    info_threshold: float = 0.60
    warning_threshold: float = 0.75
    critical_threshold: float = 0.90
    max_context_tokens: int = 128_000
    inject_role: Literal["system", "user"] = "user"
    nudge_role: Literal["system", "user"] = "user"
    nudge_visible: bool = True
    nudge_turn_frequency: int = 3
    nudge_step_frequency: int = 50
    auto_dedup: bool = True
    auto_strategy_threshold: WatermarkLevel = WatermarkLevel.INFO
    purge_error_steps: int = 3
    step_protection: int = 2
    protected_tool_patterns: tuple[str, ...] = ("ask", "confirm", "approval_*")
    protected_tools: set[str] = Field(default_factory=set)
    meta_tool_retention: int = 1
    clear_thinking_enabled: bool = False

    @field_validator("auto_strategy_threshold", mode="before")
    @classmethod
    def _coerce_watermark_level(cls, v: Any) -> Any:
        """Coerce str/int to ``WatermarkLevel`` for YAML compatibility."""
        if isinstance(v, str):
            return WatermarkLevel[v.upper()]
        if isinstance(v, int) and not isinstance(v, WatermarkLevel):
            return WatermarkLevel(v)
        return v

    @model_validator(mode="after")
    def _validate_and_expand(self) -> DCPConfig:
        """Validate thresholds and expand glob patterns into protected_tools."""
        # Threshold ordering validation.
        if self.info_threshold >= self.warning_threshold:
            msg = (
                f"info_threshold ({self.info_threshold}) must be less than "
                f"warning_threshold ({self.warning_threshold})"
            )
            raise ValueError(msg)
        if self.warning_threshold >= self.critical_threshold:
            msg = (
                f"warning_threshold ({self.warning_threshold}) must be less than "
                f"critical_threshold ({self.critical_threshold})"
            )
            raise ValueError(msg)

        # Range validation.
        if self.max_context_tokens <= 0:
            msg = f"max_context_tokens ({self.max_context_tokens}) must be positive"
            raise ValueError(msg)

        # Expand glob patterns into protected_tools.
        # User-specified entries are preserved; pattern strings are added.
        self.protected_tools |= set(self.protected_tool_patterns)

        return self
