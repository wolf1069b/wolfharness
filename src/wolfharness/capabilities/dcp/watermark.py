"""Watermark state machine for context pressure escalation.

Tracks token usage ratio and escalates through watermark levels
to drive context pruning decisions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wolfharness.capabilities.dcp.state import WatermarkLevel
from wolfharness.capabilities.dcp.token_utils import estimate_tokens


if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage


class WatermarkStateMachine:
    """Tracks context pressure and escalates watermark levels.

    Thresholds are configurable so that different deployments can tune
    the escalation points without modifying code.

    Attributes:
        last_ratio: Token ratio from the most recent ``update`` call.
        last_token_count: Estimated token count from the most recent
            ``update`` call.
        info_threshold: Pressure ratio at which INFO level activates.
        warning_threshold: Pressure ratio at which WARNING level activates.
        critical_threshold: Pressure ratio at which CRITICAL level activates.
    """

    def __init__(
        self,
        info_threshold: float = 0.60,
        warning_threshold: float = 0.75,
        critical_threshold: float = 0.90,
    ) -> None:
        """Initialise with configurable thresholds.

        Args:
            info_threshold: Pressure ratio for INFO level (default 0.60).
            warning_threshold: Pressure ratio for WARNING level (default 0.75).
            critical_threshold: Pressure ratio for CRITICAL level (default 0.90).
        """
        self.last_ratio: float = 0.0
        self.last_token_count: int = 0
        self.info_threshold: float = info_threshold
        self.warning_threshold: float = warning_threshold
        self.critical_threshold: float = critical_threshold

    def update(self, messages: list[ModelMessage], max_tokens: int) -> WatermarkLevel:
        """Estimate tokens, compute ratio, return appropriate watermark level.

        Args:
            messages: The conversation messages to evaluate.
            max_tokens: The maximum allowed token count.

        Returns:
            The watermark level corresponding to the current pressure.

        Raises:
            ValueError: If ``max_tokens`` is zero or negative.
        """
        if max_tokens <= 0:
            msg = f"max_tokens must be positive, got {max_tokens}"
            raise ValueError(msg)

        estimated = estimate_tokens(messages)
        return self.update_with_tokens(estimated, max_tokens)

    def update_with_tokens(self, token_count: int, max_tokens: int) -> WatermarkLevel:
        """Compute watermark from a pre-computed token count.

        Use this when the caller has already estimated (and optionally
        calibrated) the token count, to avoid redundant estimation.

        Args:
            token_count: Pre-computed token count (may be calibrated).
            max_tokens: The maximum allowed token count.

        Returns:
            The watermark level corresponding to the current pressure.

        Raises:
            ValueError: If ``max_tokens`` is zero or negative.
        """
        if max_tokens <= 0:
            msg = f"max_tokens must be positive, got {max_tokens}"
            raise ValueError(msg)

        self.last_token_count = token_count
        self.last_ratio = token_count / max_tokens

        if self.last_ratio >= self.critical_threshold:
            return WatermarkLevel.CRITICAL
        if self.last_ratio >= self.warning_threshold:
            return WatermarkLevel.WARNING
        if self.last_ratio >= self.info_threshold:
            return WatermarkLevel.INFO
        return WatermarkLevel.NORMAL
