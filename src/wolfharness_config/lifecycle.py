"""Lifecycle configuration model for agents.

Defines the ``LifecycleConfig`` Pydantic model that controls how the
RunLoop's six dimensions (Journal, SnapshotStore, etc.) are created.
When attached to a ``BaseAgentConfig``, it determines whether the
agent uses in-memory (default) or durable (SQLite-backed) persistence
for its lifecycle state.

Example YAML::

    agents:
      my_agent:
        type: native
        model: openai:gpt-4o
        lifecycle:
          journal: durable
          snapshot: durable
          recover_strategy: retry
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field
from schemez import Schema


RecoverStrategy = Literal["mark_interrupted", "retry"]
"""Crash recovery strategy:

- ``"mark_interrupted"``: Mark the in-flight Turn as interrupted and
  continue from the next idle state. The Turn's partial output is
  preserved in the journal but not re-executed.
- ``"retry"``: Re-execute the interrupted Turn from the beginning,
  skipping tools that already completed (via journal tool-execution
  log idempotency).
"""

StorageBackend = Literal["memory", "durable"]
"""Storage backend selector:

- ``"memory"``: In-process implementation (``MemoryJournal``,
  ``MemorySnapshotStore``). Data is lost on process exit.
- ``"durable"``: SQLite-backed implementation (``DurableJournal``,
  ``DurableSnapshotStore``). Data persists across process restarts.
"""


class LifecycleConfig(Schema):
    """Configuration for the RunLoop lifecycle dimensions.

    Controls the storage backends and crash-recovery strategy for
    an agent's RunLoop. When omitted from agent config, all defaults
    (in-memory) are used.

    Attributes:
        journal: Storage backend for the Journal dimension.
            ``"memory"`` (default) uses ``MemoryJournal``;
            ``"durable"`` uses ``DurableJournal`` with a SQLite
            database file.
        snapshot: Storage backend for the SnapshotStore dimension.
            ``"memory"`` (default) uses ``MemorySnapshotStore``;
            ``"durable"`` uses ``DurableSnapshotStore`` with a
            SQLite database file.
        recover_strategy: Strategy for recovering from a crash
            detected via the Journal's ``resume()`` method.
            ``"mark_interrupted"`` (default) marks the Turn as
            interrupted; ``"retry"`` re-executes it with tool
            idempotency.
    """

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "x-icon": "octicon:infinity-16",
            "x-doc-title": "Lifecycle Configuration",
        },
    )

    journal: StorageBackend = Field(
        default="memory",
        title="Journal storage backend",
        examples=["memory", "durable"],
    )
    """Storage backend for the Journal dimension."""

    snapshot: StorageBackend = Field(
        default="memory",
        title="Snapshot storage backend",
        examples=["memory", "durable"],
    )
    """Storage backend for the SnapshotStore dimension."""

    recover_strategy: RecoverStrategy = Field(
        default="mark_interrupted",
        title="Crash recovery strategy",
        examples=["mark_interrupted", "retry"],
    )
    """Strategy for recovering from a detected crash."""

    def is_all_defaults(self) -> bool:
        """Check if all fields are at their default values.

        Returns:
            ``True`` if journal="memory", snapshot="memory", and
            recover_strategy="mark_interrupted".
        """
        return (
            self.journal == "memory"
            and self.snapshot == "memory"
            and self.recover_strategy == "mark_interrupted"
        )


__all__ = ["LifecycleConfig", "RecoverStrategy", "StorageBackend"]
