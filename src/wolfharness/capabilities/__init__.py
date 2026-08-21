"""pdai Capability implementations for AgentPool.

Phase 6 of the thin-wrapper refactor. Each capability is a composable
agent extension using pydantic-ai's native Capability API.

Capabilities fire hooks only when ``RunExecutor.next(node)`` is called
explicitly (Phase 2 unifies all run paths to use RunExecutor).
"""

from wolfharness.capabilities.tool_schema_overlap_capability import (
    SchemaOverrideToolset,
    ToolSchemaOverlapCapability,
)
from wolfharness.capabilities.tool_schema_overlap_config import ToolSchemaOverlapConfig

__all__ = [
    "SchemaOverrideToolset",
    "ToolSchemaOverlapCapability",
    "ToolSchemaOverlapConfig",
]
