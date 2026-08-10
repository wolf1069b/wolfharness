"""Dynamic Context Pruning (DCP) capability — model-driven context window management.

Provides ``DynamicContextPruningCapability`` and ``DCPConfig`` for context
management with prune/distill/decompress tools and a 4-level watermark
escalation system.
"""

from __future__ import annotations

from wolfharness.capabilities.dcp.capability import DynamicContextPruningCapability
from wolfharness.capabilities.dcp.config import DCPConfig
from wolfharness.capabilities.dcp.state import (
    CompressionBlock,
    DCPState,
    WatermarkLevel,
)

__all__ = [
    "CompressionBlock",
    "DCPConfig",
    "DCPState",
    "DynamicContextPruningCapability",
    "WatermarkLevel",
]
