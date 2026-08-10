"""Background task types."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    from collections.abc import Callable

    from wolfharness.capabilities.background_task.manager import BackgroundTaskManager
    from wolfharness.capabilities.background_task.notification import NotificationBatcher

TaskStatus = Literal[
    "pending",
    "running",
    "cancelling",
    "completed",
    "error",
    "cancelled",
    "timed_out",
]


@dataclass(slots=True)
class BackgroundTask:
    """Serializable representation of a background task.

    Stores task metadata and lifecycle state. Does not hold any
    asyncio primitives — use `TaskHandle` for runtime references.
    """

    id: str
    description: str
    agent_or_team: str
    prompt: str
    parent_session_id: str | None
    child_session_id: str | None
    load_skills: list[str] = field(default_factory=list)
    status: TaskStatus = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    output_file: str | None = None
    result: str | None = None
    error: str | None = None


@dataclass(slots=True)
class TaskHandle:
    """Runtime handle for a background task.

    Wraps the `asyncio.Task` and provides a `completion_event` that
    is set when the task reaches a terminal state, enabling blocking
    waits via `background_output(block=True)`.

    ``blocking_waiter_id`` tracks whether a ``background_output(block=True)``
    call is actively waiting on this task.  The completion path inspects
    this flag to decide whether to inject a lead-agent notification
    (only when no blocker is present).

    ``on_completed`` is an optional callback invoked by the manager
    after the task reaches a terminal state but **before** the
    ``completion_event`` is set.  This gives the callback a chance to
    act (e.g., inject a prompt into the lead agent) while the blocking
    waiter is still registered, so it can skip the injection when
    appropriate.
    """

    task: asyncio.Task[None]
    completion_event: asyncio.Event = field(default_factory=asyncio.Event)
    blocking_waiter_id: str | None = None
    on_completed: Callable[[], None] | None = None


@dataclass(slots=True)
class SessionTaskState:
    """Per-session runtime state for background task capability.

    Stores session-scoped references to the ``BackgroundTaskManager``,
    ``NotificationBatcher``, and transient tracking fields.  Each run
    gets its own instance, keyed by ``AgentRunContext.run_id`` (a stable
    UUID string) in a plain ``dict``.
    """

    task_manager: BackgroundTaskManager
    batcher: NotificationBatcher
    pending_retrievals: set[str] = field(default_factory=set)
    retrieval_retry_count: int = 0
