"""Graph configuration models for agent workflow definitions.

These Pydantic models define the structure for graph-based agent workflows
in YAML configuration. They are used by the ``graph:`` section of AgentPool
configs and by the legacy translation layer in ``graph_translation.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

from pydantic import ConfigDict, Field, ImportString, model_serializer
from schemez import Schema

from wolfharness_config.conditions import Condition
from wolfharness_config.mcp_server import MCPServerConfig


class GraphStepConfig(Schema):
    """Configuration for a single step (node) in a graph.

    When translated from legacy ``teams:`` YAML, the following fields carry
    per-member team configuration that has no native ``graph:`` equivalent:

    - ``shared_prompt``: team-level shared prompt injected into this step
    - ``prompt_template``: Jinja2 template for per-member prompt rendering
    - ``member_timeout``: maximum seconds this step may run before cancellation
    - ``member_retry_attempts``: number of retry attempts on failure
    - ``member_retry_delay``: delay between retry attempts in seconds
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={"title": "Graph Step Configuration"},
    )

    id: str = Field(title="Step identifier")
    """Unique identifier for this step within the graph."""

    agent: str = Field(title="Agent name")
    """Name of the agent to execute at this step."""

    label: str | None = Field(default=None, title="Human-readable label")
    """Optional display label for this step."""

    mcp_servers: list[str | MCPServerConfig] = Field(default_factory=list)
    """MCP servers available to this step."""

    shared_prompt: str | None = Field(default=None, title="Shared prompt")
    """Optional shared prompt injected into this step (from team config)."""

    prompt_template: str | None = Field(default=None, title="Jinja2 prompt template")
    """Optional Jinja2 template for per-step prompt rendering (from team member config)."""

    member_timeout: float | None = Field(default=None, title="Per-step timeout (seconds)")
    """Maximum seconds this step may run before being cancelled.

    When set, steps that exceed this deadline are cancelled and their
    errors are recorded.  Other steps that finish in time are not affected.
    ``None`` (default) means no timeout.
    """

    member_retry_attempts: int = Field(default=0, title="Retry attempts")
    """Number of retry attempts on step failure (0 = no retries)."""

    member_retry_delay: float = Field(default=0.0, title="Retry delay (seconds)")
    """Delay between retry attempts in seconds."""


class GraphJoinConfig(Schema):
    """Configuration for an explicit join node in a graph."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={"title": "Graph Join Configuration"},
    )

    id: str = Field(title="Join identifier")
    """Unique identifier for this join node."""

    inputs: list[str] = Field(title="Input step IDs")
    """Step IDs whose outputs should be joined."""

    reducer: ImportString[Callable[..., Any]] | None = Field(default=None, title="Reducer function")
    """Optional import path to a reducer callable."""

    initial: Any = Field(default=None, title="Initial accumulator value")
    """Initial value for the join accumulator."""


class GraphEdgeConfig(Schema):
    """Configuration for an edge between steps in a graph."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={"title": "Graph Edge Configuration"},
    )

    from_: str | list[str] = Field(alias="from", title="Source step ID(s)")
    """ID of the step where this edge originates, or a list of IDs for an
    implicit ``Join``."""

    to: str | list[str] = Field(title="Target step ID(s)")
    """ID(s) of the step(s) this edge connects to.  A list creates a ``Fork``."""

    label: str | None = Field(default=None, title="Edge label")
    """Optional human-readable label."""

    condition: Condition | None = Field(default=None, title="Filter condition")
    """Condition for conditional routing (translated from ``filter_condition``)."""

    stop_condition: Condition | None = Field(default=None, title="Stop condition")
    """Condition that stops / disconnects this edge."""

    transform: ImportString[Callable[..., Any]] | None = Field(
        default=None, title="Transform function"
    )
    """Optional function to transform data flowing across this edge."""

    mode: str = Field(default="run", title="Connection mode")
    """How messages are handled.  One of ``run``, ``context``, or ``forward``."""

    async_: bool = Field(default=False, alias="async", title="Async execution")
    """Whether the edge executes asynchronously (does not wait for completion)."""

    map: bool = Field(default=False, title="Map fan-out")
    """Whether to fan out iterable outputs across parallel paths."""

    join: bool = Field(default=False, title="Join fan-in")
    """Whether to collect all results before continuing."""

    priority: int = Field(default=0, title="Priority")
    """Task priority (lower = higher priority)."""

    delay: timedelta | None = Field(default=None, title="Delay")
    """Optional delay before processing."""

    @model_serializer(mode="wrap")
    def _serialize(self, serializer: Callable[[Any], dict[str, Any]], info: Any) -> dict[str, Any]:
        """Serialize while preserving field aliases.

        ``schemez.Schema`` overrides the default serializer with a custom
        field-ordering implementation that does not account for aliases.
        This override ensures ``by_alias=True`` works correctly for edges.
        """
        return serializer(self)


class GraphConfig(Schema):
    """Top-level graph configuration."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={"title": "Graph Configuration"},
    )

    name: str | None = Field(default=None, title="Graph name")
    """Optional name for this graph."""

    steps: list[GraphStepConfig] = Field(default_factory=list)
    """Steps (nodes) in the graph."""

    edges: list[GraphEdgeConfig] = Field(default_factory=list)
    """Edges connecting steps in the graph."""

    joins: list[GraphJoinConfig] = Field(default_factory=list)
    """Explicit join configurations."""

    @model_serializer(mode="wrap")
    def _serialize(self, serializer: Callable[[Any], dict[str, Any]], info: Any) -> dict[str, Any]:
        """Serialize while preserving field aliases on nested models."""
        return serializer(self)
