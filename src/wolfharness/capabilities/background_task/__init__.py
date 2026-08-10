"""Background task capability for async task delegation.

Provides ``BackgroundTaskCapability`` and the supporting ``BackgroundTask``,
``TaskHandle``, ``SessionTaskState``, ``BackgroundTaskManager``, and
``NotificationBatcher`` types for managing background task delegation.
"""

from __future__ import annotations

from wolfharness.capabilities.background_task.capability import (
    BackgroundTaskCapability,
)
from wolfharness.capabilities.background_task.manager import (
    BackgroundTaskManager,
    TERMINAL_STATES,
)
from wolfharness.capabilities.background_task.notification import (
    NotificationBatcher,
)
from wolfharness.capabilities.background_task.types import (
    BackgroundTask,
    SessionTaskState,
    TaskHandle,
    TaskStatus,
)

__all__ = [
    "TERMINAL_STATES",
    "BackgroundTask",
    "BackgroundTaskCapability",
    "BackgroundTaskManager",
    "NotificationBatcher",
    "SessionTaskState",
    "TaskHandle",
    "TaskStatus",
]
